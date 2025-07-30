import torch,scipy,math,time,sys,random
import torch.nn as nn
import numpy as np
import itertools
from torch.func import jacrev, functional_call
torch.set_default_dtype(torch.float64)

def main():    
    task = IDEsolve()
    

    
class IDEsolve:
    def __init__(self,
                 num_sample      = 2**9,
                 num_subsample   = 2**9,
                 num_collocation = 2**6,
                 amin = 1.2,
                 amax = 6.0,
                 anum = 16,
                 gmin = 1.2,
                 gmax = 6.0,
                 gnum = 16,                 
                 glorder = 16,
                 node  = 128,
                 layer = 1,
                 iter_lm = 10000,
                 iter_bfgs = 10000,
                 switch_tol = 1e-4,
                 seed = 123,
                 savedir = "./",
                 loguniform_sampler = True,
                 ):
        
        init = device_set(seed=seed)
        self.device = init.device
        
        
        self.glorder = glorder
        w = np.polynomial.legendre.leggauss(self.glorder)
        self.glpts, self.glwts = torch.tensor(w[0]), torch.tensor(w[1])
        
        self.hpi = torch.tensor(0.5*np.pi)
        
        self.num_sample = num_sample
        self.num_subsample = num_subsample
        self.num_collocation = num_collocation
        
        self.num_sample = self.num_sample//self.glorder
        self.num_subsample = self.num_subsample//self.glorder
        
        self.model_pdf   = NNpdf(node, layer)
        self.model_scale = NNscale(node, layer)
        
        
        vals = [self.num_sample,self.glorder,self.num_sample*self.glorder]
        print(" Grids for norm: {:d}x{:1d} = {:d}".format(*vals))
        vals = [self.num_subsample,self.glorder,self.num_subsample*self.glorder]
        print(" Grids for kern: {:d}x{:1d} = {:d}".format(*vals))
        print(" Grids for IDE : {:d}".format(self.num_collocation))
        
        
        self.models = [self.model_pdf, self.model_scale]        
        self.model_parameters = list(self.model_pdf.parameters()) + list(self.model_scale.parameters())        
        self.get_parameter_information(self.models)        
        self.save_path = savedir + "weight"



        def uniform_density(x,y):
            return 1.0/(amax-amin)/(gmax-gmin)
        def loguniform_density(x,y):
            return 1.0/x/y/(math.log(amax)-math.log(amin))/(math.log(gmax)-math.log(gmin))
        
        with torch.no_grad():
            if loguniform_sampler:
                print(" Log uniform integration for parameter space.")
                aa = torch.exp(torch.linspace(math.log(amin),math.log(amax),anum))
                gg = torch.exp(torch.linspace(math.log(gmin),math.log(gmax),gnum))
                func = loguniform_density
            else:                
                print(" Uniform integration for parameter space.")
                aa = torch.linspace(amin,amax,anum)
                gg = torch.linspace(gmin,gmax,gnum)        
                func = uniform_density
                
            self.params_pts, self.params_wts = self.compute_weights(aa,gg,5,f=func)        
        
        aa = self.num_collocation+2
        bb = self.params_pts.shape[0]
        print(" Residuals : {:d}x{:d}={:d}".format(aa,bb,aa*bb))

        self.lossfile = savedir + "loss.txt"
        self.iter_lm, self.iter_bfgs = iter_lm, iter_bfgs        
        self.switch_tol = switch_tol        
        self.fev = 0
        self.timer = stopwatch()
        self.train()
        return




    def train(self,):                
        with open(self.lossfile,mode="w") as ff:
            print("#","step, loss, test21, test22, time",file=ff)
        steps = 0
        tol   = self.switch_tol
        if self.iter_bfgs > 0:
            steps = self.train_bfgs(start_step=steps,tol=tol)
            self.load_model()
        if self.iter_lm > 0:
            steps = self.train_lm(start_step=steps,tol=tol)
        return

    
    
    
    def train_lm(self,start_step=0,tol=1e-4):
        self.wrap  = ResidualWrapper(self.model_pdf, self.model_scale)        
        
        steps = start_step
        start_time = time.time()
        
        lmlam = tol
        inc = 1.2
        dec = 0.5        
        eta = 1.0        
        c1 = 1e-4
        for _ in range(self.iter_lm):
            steps += 1
            self.fev += 1
            t0 = self.wrap.theta0.clone().detach()
            
            l,jacob,Gramian,R  = self.loss()
            l = l.item()

            d,gd = self.get_decent(min([lmlam,l]),Gramian,R,jacob)

            del jacob,Gramian,R
            torch.cuda.empty_cache()
            
            def obj(t):
                t1 = t0 + t * d
                self.wrap.load_vector_(t1)
                l1 = self.loss(loss_only=True)
                if torch.isnan(l1) or l1 > 1e30: return 1e30
                l1 = l1.item()
                return l1
            
            if gd >= 0.0 :
                print(" Not a decent direction. Terminated.")
                break
            if eta < 1e-30:
                print(" Converged to a local. Terminated.")
                break
            
            
            for _ in range(1000):
                l1 = obj(eta)
                print(" Armijo: curr {:.5e}, cand {:.5e}, lr {:.5e}".format(l,l1,eta))
                if l1 < l + c1 * eta * gd.item():
                    eta *= inc
                    break
                else:
                    eta *= dec
                    if eta < 1e-30: break                    
                
                    
            self.save_model()            
            vals = [self.fev, l1, self.timer.count()]
            txt = "LM: {:10d}, Loss: {:.4e}, time: {:.1f},"
            print(txt.format(*vals))
            
            with open(self.lossfile,mode="a") as ff:
                test_loss = self.get_testloss()
                vals = [steps,l1,*test_loss,self.timer.elapsed()]
                txts = " {:d} {:.5e}" + " {:.5e}" * len(test_loss) + " {:.1f}"
                print(txts.format(*vals),file=ff)            
        return steps
    
    def train_bfgs(self,start_step=0,tol=1e-4):
        steps = start_step
        optimizer = torch.optim.LBFGS(self.model_parameters,
                                      history_size=200,
                                      max_iter=20,
                                      tolerance_grad=0.0,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=0.0)        
        latest_loss = [None]
        global_state = None        
        for epoch in range(self.iter_bfgs):
            start_time = time.time()
            def closure():
                self.fev += 1
                optimizer.zero_grad()
                l = self.loss(loss_only=True,calc_grad=True)
                latest_loss[0] = l.detach().item()
                vals = [self.fev, l.item(),self.timer.count()]
                txt = "LBFGS: {:10d}, Loss: {:.4e}, time: {:.1f},"
                print(txt.format(*vals))
                return l
            
            if optimizer.state:
                if global_state is None:
                    global_state = optimizer.state[next(iter(optimizer.state))]
                n_before = global_state.get('n_iter', 0)
            else:
                n_before = 0
                
            optimizer.step(closure)
            
            if global_state is None:
                global_state = optimizer.state[next(iter(optimizer.state))]
            n_after      = global_state['n_iter']
            inner_iters  = n_after - n_before
            total_iters  = n_after                 
            
            steps += inner_iters
            
            self.save_model()
            
            loss = latest_loss[0]
            with open(self.lossfile,mode="a") as ff:
                test_loss = self.get_testloss()
                vals = [steps,loss,*test_loss,self.timer.elapsed()]
                txts = " {:d} {:.5e}" + " {:.5e}" * len(test_loss) + " {:.1f}"                
                print(txts.format(*vals),file=ff)
            if loss < tol : break
        return steps
    


    def get_testloss(self,):
        x,dx = self.zero_inf_sampler(self.num_sample)
        
        params = torch.tensor([2.0,1.0])        
        y1 = self.model_pdf(x,params)
        
        params = torch.tensor([2.0,2.0])        
        y2 = self.model_pdf(x,params)
        
        y1 = y1.flatten().to('cpu').detach().numpy().copy()
        y2 = y2.flatten().to('cpu').detach().numpy().copy()
        x = x.flatten().to('cpu').detach().numpy().copy()
        dx = dx.flatten().to('cpu').detach().numpy().copy()
        
        e2 = scipy.special.expn(2,1.2*x)
        e3 = scipy.special.expn(3,1.2*x)
        z1 = 432.0/25.0 * x * (e2-e3).clip(min=0.0)

        q1 = 27.0*np.pi/32.0 * x * np.exp(-9.0*np.pi/64.0 * x*x)
        q2 = 81.0*np.pi*np.pi/256.0 * x*x * scipy.special.erfc(3.0*np.sqrt(np.pi)/8.0 * x)
        z2 = (q1 - q2).clip(min=0.0)
        
        d1 = np.square(y1-z1)
        d2 = np.square(y2-z2)
        
        l1 = (d1*dx).sum()
        l2 = (d2*dx).sum()
        return l1,l2
        

    @torch.no_grad()
    def compute_weights(self,X, Y, h, f=None):
        if f is None:
            def func(x,y): return 1.0
        else:
            func = f
        
        n,m = X.shape[0],Y.shape[0]
        if n < 2 or m < 2:
            raise ValueError("Need at least two grid points in each direction.")        
        if not (torch.all(X[1:] > X[:-1]) and torch.all(Y[1:] > Y[:-1])):
            raise ValueError("X and Y must be strictly increasing.")
        z_np, w_np = np.polynomial.legendre.leggauss(h)
        z = torch.from_numpy(z_np).to(dtype=X.dtype, device=X.device)
        w = torch.from_numpy(w_np).to(dtype=X.dtype, device=X.device)

        s_nodes = 0.5 * (1.0 + z)
        t_nodes = s_nodes.clone()
        
        N = torch.empty((h, h, 4), dtype=X.dtype, device=X.device)
        for a in range(h):
            s = s_nodes[a]
            for b in range(h):
                t = t_nodes[b]
                N[a, b, 0] = (1.0 - s) * (1.0 - t)
                N[a, b, 1] = s * (1.0 - t)
                N[a, b, 2] = (1.0 - s) * t
                N[a, b, 3] = s * t

        base_weight = torch.outer(w, w) / 4.0 
        
        Z1 = torch.zeros((n, m), dtype=X.dtype, device=X.device)
        Z2 = torch.zeros((n, m), dtype=X.dtype, device=X.device)
                
        for k in range(n):
            x0 = X[k]
            for l in range(m):
                y0 = Y[l]
                Z1[k,l] += x0
                Z2[k,l] += y0
        Z = torch.cat([Z1.flatten().unsqueeze(-1),Z2.flatten().unsqueeze(-1)],dim=-1)
        
        W = torch.zeros((n, m), dtype=X.dtype, device=X.device)
        for k in range(n - 1):
            x0 = X[k]
            dx   = X[k + 1] - X[k] 
            for l in range(m - 1):
                y0 = Y[l]
                dy = Y[l + 1] - Y[l]
                
                x_gp = x0 + s_nodes.view(h, 1) * dx
                y_gp = y0 + t_nodes.view(1, h) * dy
                
                f_gp = func(x_gp, y_gp)

                jw = base_weight * dx * dy
                
                contrib = (jw * f_gp).unsqueeze(-1) * N
                c00 = contrib[:, :, 0].sum()
                c10 = contrib[:, :, 1].sum()
                c01 = contrib[:, :, 2].sum()
                c11 = contrib[:, :, 3].sum()
                
                W[k    , l    ] += c00
                W[k + 1, l    ] += c10
                W[k    , l + 1] += c01
                W[k + 1, l + 1] += c11
        W = W.flatten()
        return Z.clone().detach(), W.clone().detach()    
        
    

    
    
       
    def get_parameter_information(self,model_list):
        total_trainables = 0
        for i,model in enumerate(model_list):
            trainable = 0
            nontrainable = 0
            for n,p in model.named_parameters():
                if p.requires_grad :
                    trainable = trainable + p.data.numel()
                else:
                    nontrainable = nontrainable + p.data.numel()
            print("------ "+"MODEL "+str(i)+" info"+" ------")
            if trainable >0 : print("  Trainable : {:,}".format(trainable))
            if nontrainable > 0: print("  Non-trainable : {:,}".format(nontrainable))
            print("")
            total_trainables = total_trainables + trainable
        if len(model_list)>1 :
            print("Total trainables : {:,}".format(total_trainables))
        return

    def save_model(self,):
        for i,model in enumerate(self.models):
            torch.save(model.state_dict(),self.save_path+str(i+1)+".pth")
        return

    def load_model(self,):
        for i,model in enumerate(self.models):        
            model.load_state_dict(torch.load(self.save_path+str(i+1)+".pth",weights_only=True))
        print(" model loaded.")
        return




    
    @torch.no_grad()
    def beta_distribution(self,x,alpha):
        y = torch.where( (x > 0) & (x < 1) , torch.pow(x*(1.0-x),alpha-1.0),0)
        c = torch.lgamma(torch.tensor([alpha,2.0*alpha]))
        return y * torch.exp(c[1]-2*c[0])

    @torch.no_grad()
    def zero_inf_sampler(self,num):
        t,dt = self.glpoints(num,tmin=-3.0,tmax=2.0)
        q = self.hpi * torch.sinh(t)
        z = torch.exp(q)
        dz = z * self.hpi * torch.cosh(t) * dt
        return z,dz

    @torch.no_grad()    
    def zero_one_sampler(self,num):
        t,dt = self.glpoints(num,tmin=-3.0,tmax=3.0)
        q = self.hpi * torch.sinh(t)
        z = 0.5 * torch.tanh(q) + 0.5
        dz = 0.5 * self.hpi * torch.cosh(t) / torch.square(torch.cosh(q)) * dt
        return z,dz

    @torch.no_grad()
    def zero_inf_sampler_regular(self,num):
        tmin,tmax = -3.0,2.0
        t = torch.linspace(tmin,tmax,num).reshape(-1,1)
        dt = torch.tensor((tmax - tmin)/(num-1))
        q = self.hpi * torch.sinh(t)
        z = torch.exp(q)
        dz = z * self.hpi * torch.cosh(t) * dt        
        return z,dz
    
    


    
    
    @torch.no_grad()
    def glpoints(self,num,tmin,tmax):
        T = torch.linspace(tmin,tmax,num+1)
        dt = (tmax - tmin )/num
        t = ((self.glpts * dt)[None,:] + (T[:-1]+T[1:])[:,None])*0.5
        w = (self.glwts * dt * 0.5)[None,:] * torch.ones(num)[:,None]
        return t.reshape(-1,1), w.reshape(-1,1)


    def loss(self,lmth=1e-4,loss_only=False,calc_grad=False,vmax=1e99):
        nparam_samples = self.params_pts.shape[0]
        
        if loss_only :
            L = torch.tensor(0.0,requires_grad=False)
            for param,w in zip(self.params_pts,self.params_wts):
                R = self.get_residuals(self.model_pdf,self.model_scale,param)
                ls = torch.square(R) * 0.5
                l = ls.sum() * w
                if calc_grad : l.backward()
                L = L + l.clone().detach()
            if torch.isnan(L) or L > 1e30:
                L = torch.tensor(1e30)
            aa = torch.nn.utils.clip_grad_norm_(self.model_parameters,max_norm=vmax)
            return L
        
        
        L,R,J = torch.tensor(0.0,requires_grad=False), [], []
        for i,(param,w) in enumerate(zip(self.params_pts,self.params_wts)):
            sqw = torch.sqrt(w)
            
            self.wrap.theta0 = self.wrap.theta0.detach().requires_grad_(True)
            l, g, jac, r = self.wrap.get_jacobian(self.get_residuals,param)
            
            L = L + w * l.clone().detach() 
            
            R.append(sqw *   r.clone().detach())
            J.append(sqw * jac.clone().detach())
            
            del l, g, r, jac
            torch.cuda.empty_cache()

        with torch.no_grad():
            J = torch.cat(J,dim=0)
            R = torch.cat(R,dim=0)
            A = torch.matmul(J,torch.t(J))
        return L.clone().detach(), J.clone().detach(), A.clone().detach(), R.clone().detach()
    
    def get_decent(self,lmlam,Gramian,R,jacob):
        G = torch.t(jacob) @ R
        B = -jacob @ G
        y = torch.linalg.solve(Gramian+torch.eye(Gramian.shape[0])*lmlam, B)
        d = (torch.mv(torch.t(jacob),y) + G) * (-1.0/lmlam)
        return d, G @ d



    
    
    def normalization_residual(self,model_pdf,params):
        x,dx = self.zero_inf_sampler(self.num_sample)
        y = model_pdf(x,params)        
        mean = torch.sum(y * x * dx)
        norm = torch.sum(y * dx)        
        return mean - 1.0, norm - 1.0
    
    def IDE_residual(self,model_pdf,model_scale,params):
        x,dx = self.zero_inf_sampler_regular(self.num_collocation)
        x.requires_grad = True
        
        
        alpha,gamma = params[0], params[1]
        scale = model_scale(params.unsqueeze(0))        
        
        y = model_pdf(x,params)    
        
        gg = torch.autograd.grad(x * y, x,
                                 grad_outputs=torch.ones_like(y),
                                 create_graph=True,retain_graph=True,allow_unused=False)[0]
        f1 = (1.0/gamma) * gg


        x.requires_grad = False
        f2 = torch.pow(x/scale,gamma) * y

        with torch.no_grad():
            z,dz = self.zero_one_sampler(self.num_subsample)
            df3 = dz/z * self.beta_distribution(z,alpha)
            u = x.flatten()[:,None] / z.flatten()[None,:]
            v = u.reshape(-1,1)        

        yu = model_pdf(v,params)        
        f3 = torch.sum(df3.flatten()[None,:] * torch.pow(u/scale,gamma) * yu.reshape(u.shape), dim=1).reshape(-1,1)

        weight = 1.0/torch.sqrt(x)
        return (f1+f2-f3) * torch.sqrt(dx) * weight 
    
    

    def get_residuals(self,model_pdf,model_scale,params):
        r1,r2 = self.normalization_residual(model_pdf,params)
        r3 = self.IDE_residual(model_pdf,model_scale,params)
        return torch.cat([r1.unsqueeze(0).unsqueeze(0),
                          r2.unsqueeze(0).unsqueeze(0),
                          r3],dim=0)
    
    








    
class ResidualWrapper(nn.Module):
    def __init__(self, pdf: nn.Module, scale: nn.Module):
        super().__init__()
        self.pdf, self.scale = pdf, scale

        self._meta, flat = [], []
        for prefix, m in [('pdf', pdf), ('scale', scale)]:
            for name, p in m.named_parameters():
                self._meta.append((prefix, name, p.shape, p.numel()))
                flat.append(p.detach().reshape(-1))
        self.theta0 = torch.cat(flat).requires_grad_(True)

        self._buf = {
            'pdf':   dict(pdf.named_buffers()),
            'scale': dict(scale.named_buffers())
        }

    @staticmethod
    def _fcall(mod, params, bufs, *args, **kw):
        state = {**params, **bufs}
        return functional_call(mod, state, args, kw)

    def _unflatten(self, vec):
        splits = vec.split([m[3] for m in self._meta])
        d_pdf, d_scale = {}, {}
        for chunk, (prefix, name, shape, _) in zip(splits, self._meta):
            (d_pdf if prefix == 'pdf' else d_scale)[name] = chunk.view(shape)
        return d_pdf, d_scale

    def _residuals(self, vec, res_fn, *a, **k):
        p_pdf, p_scale = self._unflatten(vec)

        def mc(*inputs, **kw_inputs):
            return self._fcall(self.pdf, p_pdf, self._buf['pdf'],
                                *inputs, **kw_inputs)

        def ms(*inputs, **kw_inputs):
            return self._fcall(self.scale, p_scale, self._buf['scale'],
                                *inputs, **kw_inputs)
        return res_fn(mc, ms, *a, **k).squeeze()

    @torch.no_grad()
    def load_vector_(self, vec: torch.Tensor) -> None:
        p_pdf, p_scale = self._unflatten(vec)

        for n, p in self.pdf.named_parameters():
            p.copy_(p_pdf[n])
        for n, p in self.scale.named_parameters():
            p.copy_(p_scale[n])
            
        self.theta0 = vec.detach().clone().requires_grad_(True)
    
    def get_jacobian(self, res_fn, *a, **k):
        def g(v):
            r = self._residuals(v, res_fn, *a, **k)
            return r, r
        J, R = jacrev(g, has_aux=True)(self.theta0)
        L, grad = 0.5 * R.square().sum(), J.T @ R
        return L, grad, J, R
    






class device_set:
    def __init__(self,seed=123):
        if torch.cuda.is_available():
            print("GPU loading completed.", torch.cuda.current_device())
            torch.set_default_device('cuda')
            self.device = torch.device('cuda')
        else:
            print("No GPU detected.")
            torch.set_default_device('cpu')
            self.device = torch.device('cpu')
        self.torch_fix_seed(seed=seed)
        return 
    def torch_fix_seed(self,seed=123):
        print( "seed : {:d}".format(seed))
        random.seed(seed) # Python random
        np.random.seed(seed) # Numpy
        torch.manual_seed(seed) # Pytorch
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        return














    


    
class NNpdf(nn.Module):
    def __init__(self, node, layer):
        super(NNpdf, self).__init__()
        self.init_layer = FourierFeature(3,node)
        self.dense = nn.ModuleList([ NomalLayer(node,node) for j in range(layer)])
        self.out_layer = NomalLayer(node,1 )
        self.activation = torch.nn.SiLU()
        print( " node: {:d}, layer: {:d} ".format(node,layer))
        print(self.activation)
        return
    def forward(self,x,params):
        z = torch.log(x)
        y = torch.cat([z,params.unsqueeze(0).expand(z.shape[0],-1)],dim=1)
        y = self.init_layer(y)
        
        y = self.activation(y)
        for i,fc in enumerate(self.dense):
            y = self.activation(fc(y)) + y
        y = (params[0]-1.0)*self.out_layer(y)                    
        y = torch.exp(y)
        logscale = torch.lgamma(params[0]/params[1]) - torch.lgamma((params[0]+1.0)/params[1])
        scale = torch.exp(logscale)
        norm = torch.log(params[1]) - params[0]*logscale - torch.lgamma(params[0]/params[1])
        prior = norm + (params[0]-1.0)*torch.log(x) - torch.pow(x/scale,params[1])        
        return y * torch.exp(prior)

class NNscale(nn.Module):
    def __init__(self, node, layer):
        super(NNscale, self).__init__()
        self.init_layer = FourierFeature(2,node)
        self.dense = nn.ModuleList([ NomalLayer(node,node) for j in range(layer)])
        self.out_layer = NomalLayer(node,1 )
        self.activation = torch.nn.SiLU()
        print( " node: {:d}, layer: {:d} ".format(node,layer))
        print(self.activation)        
        return
    def forward(self,params):
        t = self.init_layer(params)
        t = self.activation(t)
        for i,fc in enumerate(self.dense):
            t = self.activation(fc(t)) + t
        t = (params[:,0]-1.0)*self.out_layer(t)
        t = torch.exp(t)        
        logscale = torch.lgamma(params[:,0]/params[:,1]) - torch.lgamma((params[:,0]+1.0)/params[:,1])
        scale = torch.exp(logscale)        
        return t * scale

class NomalLayer(torch.nn.Module):
    def __init__(self,in_features,out_features,std=1e-3):
        super().__init__()
        std_ = std/in_features        
        weight = torch.empty(out_features,in_features)
        torch.nn.init.normal_(weight,std=std_)        
        self.weight = torch.nn.Parameter(weight,requires_grad=True)
        bias = torch.empty(out_features)
        torch.nn.init.zeros_(bias)
        self.bias = torch.nn.Parameter(bias,requires_grad=True)
        fctr = torch.empty(out_features)
        torch.nn.init.zeros_(fctr)
        self.fctr = torch.nn.Parameter(fctr,requires_grad=True)
    def forward(self,x):
        s = torch.exp(self.fctr)
        return torch.nn.functional.linear(x,s[:,None]*self.weight, bias = s*self.bias)
        

class FourierFeature(torch.nn.Module):
    def __init__(self,in_features,out_features,std=1.0):
        super().__init__()
        weight = torch.empty(out_features//2,in_features)
        torch.nn.init.normal_(weight,std=std)        
        self.weight = torch.nn.Parameter(weight,requires_grad=True)
        self.layer = NomalLayer(out_features,out_features)
    def forward(self,x):
        y = torch.nn.functional.linear(x,self.weight)
        y = torch.cat([torch.sin(y),torch.cos(y)],dim=1)
        return self.layer(y)
    
class stopwatch:
    def __init__(self,):
        self.start_time = time.time()
        self.origin = time.time()
    def count(self,):
        diff = time.time() - self.start_time
        self.start_time	= time.time()
        return diff
    def elapsed(self,):
        diff = time.time() - self.origin
        return diff
    
if __name__ == "__main__":
    main()
