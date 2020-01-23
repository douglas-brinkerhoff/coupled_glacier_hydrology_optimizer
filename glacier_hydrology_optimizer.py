import numpy as np
import matplotlib.pyplot as plt
import dolfin as df

df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 4
df.parameters['allow_extrapolation'] = True

# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

from numpy.polynomial.legendre import leggauss
def half_quad(order):
    points,weights = leggauss(order)
    points=points[(order-1)//2:]
    weights=weights[(order-1)//2:]
    weights[0] = weights[0]/2
    return points,weights

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)
def Softplus(a,b,alpha=1): return Max(a,b) + 1./alpha*df.ln(1 + df.exp(-abs(a-b)*alpha))

mesh = df.Mesh()
with df.XDMFFile("terrestrial/terrestrial.xdmf") as infile:
    infile.read(mesh)
mesh.init()

class SpecFO(object):
    def __init__(self,mesh):
        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.elements = [E_cg]*4

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=464158,p=1,q=1,beta2=2e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)
        self.g = df.Constant(g)
        self.rho_i = df.Constant(rho_i)
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.Be = df.Constant(Be)
        self.p = df.Constant(p)
        self.q = df.Constant(q)
        self.beta2 = df.Function(self.coupler.Q_r)
        self.beta2.vector()[:] = beta2

        self.ubar = self.coupler.U[0]
        self.vbar = self.coupler.U[1]
        self.udef = self.coupler.U[2]
        self.vdef = self.coupler.U[3]

        self.u0_bar = df.Function(self.coupler.Q_cg)
        self.v0_bar = df.Function(self.coupler.Q_cg)
        self.u0_def = df.Function(self.coupler.Q_cg)
        self.v0_def = df.Function(self.coupler.Q_cg)

        self.previous_time_vars = [self.u0_bar,self.v0_bar,self.u0_def,self.v0_def]

        self.lamdabar_x = self.coupler.Lambda[0]
        self.lamdabar_y = self.coupler.Lambda[1]
        self.lamdadef_x = self.coupler.Lambda[2]
        self.lamdadef_y = self.coupler.Lambda[3]

        # TEST FUNCTION COEFFICIENTS
        coef = [lambda s:1.0, lambda s:1./(n+1)*((n+2)*s**(n+1) - 1)]
        dcoef = [lambda s:0, lambda s:(n+2)*s**n]   
 
        u_ = [self.ubar,self.udef]
        v_ = [self.vbar,self.vdef]
        lamda_x_ = [self.lamdabar_x,self.lamdadef_x]
        lamda_y_ = [self.lamdabar_y,self.lamdadef_y]

        self.u = VerticalBasis(u_,coef,dcoef)
        self.v = VerticalBasis(v_,coef,dcoef)
        self.lamda_x = VerticalBasis(lamda_x_,coef,dcoef)
        self.lamda_y = VerticalBasis(lamda_y_,coef,dcoef)

        self.U_b = df.as_vector([self.u(1),self.v(1)]) 
 
        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Us = df.project(df.as_vector([self.u(0),self.v(0)]))
            self.Ub = df.project(df.as_vector([self.u(1),self.v(1)]))
            self.Us_file = df.File(results_dir+'U_s.pvd')
            self.Ub_file = df.File(results_dir+'U_b.pvd')

    def write_variables(self,t):
        Us_temp = df.project(df.as_vector([self.u(0),self.v(0)]))
        Ub_temp = df.project(df.as_vector([self.u(1),self.v(1)]))

        self.Us.vector().set_local(Us_temp.vector().get_local())
        self.Ub.vector().set_local(Ub_temp.vector().get_local())
        self.Us_file << (self.Us,t)
        self.Ub_file << (self.Ub,t)

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        p = self.p
        q = self.q
        beta2 = self.beta2

        u = self.u
        v = self.v
        lamda_x = self.lamda_x
        lamda_y = self.lamda_y
        H = self.coupler.H_c
        B = self.coupler.B_c
        S = self.coupler.S
        N = self.coupler.hydro.N

        def dsdx(s):
            return 1./H*(S.dx(0) - s*H.dx(0))

        def dsdy(s):
            return 1./H*(S.dx(1) - s*H.dx(1))

        def dsdz(s):
            return -1./H 

        # 2nd INVARIANT STRAIN RATE
        def epsilon_dot(s):
            return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                        +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                        +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                        +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                        + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                        + eps_reg)

        # VISCOSITY
        def eta_v(s):
            return Be/2.*epsilon_dot(s)**((1.-n)/(2*n))

        # MEMBRANE STRESSES
        def membrane_xx(s):
            return (lamda_x.dx(s,0) + lamda_x.ds(s)*dsdx(s))*H*(eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        def membrane_xy(s):
            return (lamda_x.dx(s,1) + lamda_x.ds(s)*dsdy(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yx(s):
            return (lamda_y.dx(s,0) + lamda_y.ds(s)*dsdx(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yy(s):
            return (lamda_y.dx(s,1) + lamda_y.ds(s)*dsdy(s))*H*(eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        # SHEAR STRESSES
        def shear_xz(s):
            return dsdz(s)**2*lamda_x.ds(s)*H*eta_v(s)*u.ds(s)

        def shear_yz(s):
            return dsdz(s)**2*lamda_y.ds(s)*H*eta_v(s)*v.ds(s)

        # DRIVING STRESSES
        def tau_dx(s):
            return rho_i*g*H*S.dx(0)*lamda_x(s)

        def tau_dy(s):
            return rho_i*g*H*S.dx(1)*lamda_y(s)

        # GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE 
        # POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
        points,weights = half_quad(9)

        # INSTANTIATE VERTICAL INTEGRATOR
        vi = VerticalIntegrator(points,weights)

        tau_bx = -beta2*Max(N,5e4)**p*abs(u(1)**2 + v(1)**2 + 1e-3)**((q-1)/2.)*u(1)
        tau_by = -beta2*Max(N,5e4)**p*abs(u(1)**2 + v(1)**2 + 1e-3)**((q-1)/2.)*v(1)

        R_u_body = (- vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) + tau_bx*lamda_x(1) - vi.intz(tau_dx))*df.dx
        R_v_body = (- vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) + tau_by*lamda_y(1) - vi.intz(tau_dy))*df.dx

        self.coupler.R += R_u_body
        self.coupler.R += R_v_body

class UnderPressure(object):
    def __init__(self,mesh):
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.elements = [E_dg]*4

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=464158,p=1,q=1,beta2=2e-3,e_v=1e-3,alpha=5./4.,beta=3./2.,k=1e5,h_r=0.1,l_r=1.0,A=2./9*1e-17,theta=1.0,dt=1e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)
        self.g = df.Constant(g)
        self.rho_i = df.Constant(rho_i)
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.Be = df.Constant(Be)
        self.p = df.Constant(p)
        self.q = df.Constant(q)
        self.beta2 = df.Constant(beta2)
        self.e_v = df.Constant(e_v)
        self.alpha = df.Constant(alpha)
        self.beta = df.Constant(beta)
        self.k = df.Constant(k)
        self.h_r = df.Constant(h_r)
        self.l_r = df.Constant(l_r)
        self.A = df.Constant(A)
        self.theta = df.Constant(theta)

        H = self.coupler.H_d
        B = self.coupler.B_d

        self.h_w = self.coupler.U[4]
        self.h = self.coupler.U[5]
        self.phi_x = self.coupler.U[6]
        self.phi_y = self.coupler.U[7]

        self.h0_w = df.Function(self.coupler.Q_dg)
        self.h0 = df.Function(self.coupler.Q_dg)
        self.phi0_x = df.Function(self.coupler.Q_dg)
        self.phi0_y = df.Function(self.coupler.Q_dg)

        self.previous_time_vars = [self.h0_w,self.h0,self.phi0_x,self.phi0_y]

        self.xsi = self.coupler.Lambda[4]
        self.chi = self.coupler.Lambda[5]
        self.w_x = self.coupler.Lambda[6]
        self.w_y = self.coupler.Lambda[7]

        self.h_mid = theta*self.h_w + (1-theta)*self.h0_w

        self.P_0 = rho_i*g*H
        self.phi_m = rho_w*g*B
        self.P_w = rho_w*g*Softplus(self.h_mid, 1./e_v*(self.h_mid - self.h) + self.h,alpha=0.01)
        self.phi = self.phi_m + self.P_w
        self.N = self.P_0 - self.P_w

        self.dt = df.Constant(dt)

        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Nout = df.project(self.N,coupler.Q_dg)
            self.phiout = df.project(self.phi,coupler.Q_dg)
            self.h_file = df.File(results_dir+'h.pvd')
            self.hw_file = df.File(results_dir+'h_w.pvd')
            self.N_file = df.File(results_dir+'N.pvd')
            self.phi_file = df.File(results_dir+'phi.pvd')
            self.phix_file = df.File(results_dir+'phix.pvd')
            self.phiy_file = df.File(results_dir+'phiy.pvd')

    def write_variables(self,t):
        N_temp = df.project(self.N,coupler.Q_dg)
        phi_temp = df.project(self.phi,coupler.Q_dg)
        self.Nout.vector().set_local(N_temp.vector().get_local())
        self.phiout.vector().set_local(phi_temp.vector().get_local())
        self.h_file << (self.h0,t)
        self.hw_file << (self.h0_w,t)
        self.N_file << (self.Nout,t)    
        self.phi_file << (self.phiout,t)
        self.phix_file << (self.phi0_x,t)
        self.phiy_file << (self.phi0_y,t)

    def set_boundary_labels(self,edgefunction):
        self.edgefunction = edgefunction
      

    def init_variables(self,f_init=0.01,h_init=0.2):
        self.h0.vector()[:] = h_init
        h0_w_temp = df.project(self.h0 + f_init*coupler.H_d*self.e_v,coupler.Q_dg)
        self.h0_w.vector()[:] = h0_w_temp.vector()[:]

    def set_timestep(self,dt):
        if self.dt:
            self.dt.assign(dt)
        else:
            self.dt = df.Constant(dt)

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        p = self.p
        q = self.q
        beta2 = self.beta2
        e_v = self.e_v
        alpha = self.alpha
        beta = self.beta
        k = self.k
        h_r = self.h_r 
        l_r = self.l_r
        A = self.A
        theta = self.theta

        normal = df.FacetNormal(mesh)
        edgefunction = self.edgefunction

        h = self.h
        h_w = self.h_w
        h_mid = self.h_mid
        phi_x = self.phi_x
        phi_y = self.phi_y

        h0 = self.h0
        h0_w = self.h0_w

        xsi = self.xsi
        chi = self.chi
        w_x = self.w_x
        w_y = self.w_y

        dt = self.dt

        m = self.coupler.m

        grad_phi = df.as_vector([phi_x,phi_y])
        w_vec = df.as_vector([w_x,w_y])

        phi_avg = 0.5*(grad_phi('+') + grad_phi('-'))
        phi_jump = df.dot(grad_phi('+'),normal('+')) + df.dot(grad_phi('-'),normal('-'))

        h_w_avg = 0.5*(Min(h_mid('+'),h('+')) + Min(h_mid('-'),h('-')))
        h_w_jump = Min(h_mid,h)('+')*normal('+') + Min(h_mid,h)('-')*normal('-')

        xsi_jump = (xsi('-')*normal('-') + xsi('+')*normal('+'))
        w_jump = df.dot(normal('+'),w_vec('+')) + df.dot(normal('-'),w_vec('-'))

        u_h = -k*(Min(h_mid,h)**2 + 1e-3)**((alpha-1)/2.)*(df.dot(grad_phi,grad_phi) + 1e0)**(beta/2.-1)*grad_phi
        u_norm = (df.dot(u_h,u_h) + 1e-5)**0.5
        u_b = (self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-3)**0.5

        #Upwind
        uH = df.avg(u_h*Min(h_mid,h)) + 0.5*df.avg(u_norm)*h_w_jump

        O = Max(u_b*(h_r - h)/l_r,0)
        C = A*h*abs(self.N)**(n-1)*self.N

        self.unsteady = df.Constant(1.0)
        R_hw = (self.unsteady*(h_w-h0_w)/dt - m )*xsi*df.dx + df.dot(uH,xsi_jump)*df.dS + xsi*df.dot(u_h*Min(h_mid,h),normal)*df.ds(subdomain_data=edgefunction)(1)# + df.dot(u_h*Min(h_mid,h),normal)*df.ds(2)
        R_gradphi = df.dot(w_vec,grad_phi)*df.dx - w_jump*df.avg(self.phi)*df.dS - df.dot(w_vec,normal)*self.phi*df.ds(subdomain_data=edgefunction)(2) - df.dot(w_vec,normal)*rho_w*g*(self.coupler.B_d)*df.ds(subdomain_data=edgefunction)(1)#_c+self.coupler.H_c)*df.ds(subdomain_data=edgefunction)(1) 

        R_h = (self.unsteady*(h - h0)/dt - O + C)*chi*df.dx

        self.coupler.R += R_hw + R_gradphi + R_h

    def init_dirichlet_bcs(self):
        self.bcs = []

class Shakti(object):
    def __init__(self,mesh):
        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.elements = [E_cg,E_dg,E_dg]

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=214000,
            p=1,q=1,beta2=2e-3,La=3.35e5,G=0.042*60**2*24*365,nu=1.787e-6*60**2*24*365,
            omega=1e-2,e_v=1e-3,ct=7.5e-8,cw = 4.22e3,alpha=5./4.,beta=3./2.,k=3e5,h_r=0.02,
            l_r=0.3,A=5.25e-25*60**2*24*365,theta=1.0,dt=1e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)
        self.g = df.Constant(g)
        self.rho_i = df.Constant(rho_i)
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.La = df.Constant(La)
        self.Be = df.Constant(Be)
        self.p = df.Constant(p)
        self.q = df.Constant(q)
        self.beta2 = df.Constant(beta2)
        self.omega = df.Constant(omega)
        self.G = df.Constant(G)
        self.nu = df.Constant(nu)
        self.e_v = df.Constant(e_v)
        self.ct = df.Constant(ct)
        self.cw = df.Constant(cw)
        self.alpha = df.Constant(alpha)
        self.beta = df.Constant(beta)
        self.k = df.Constant(k)
        self.h_r = df.Constant(h_r)
        self.l_r = df.Constant(l_r)
        self.A = df.Constant(A)
        self.theta = df.Constant(theta)

        H = self.coupler.H_c
        B = self.coupler.B_c

        self.h_w = self.coupler.U[4]
        self.h = self.coupler.U[5]
        self.K = self.coupler.U[6]

        self.h0_w = df.Function(self.coupler.Q_cg)
        self.h0 = df.Function(self.coupler.Q_dg)
        self.K0 = df.Function(self.coupler.Q_dg)

        self.previous_time_vars = [self.h0_w,self.h0,self.K0]

        self.xsi = self.coupler.Lambda[4]
        self.psi = self.coupler.Lambda[5]
        self.w = self.coupler.Lambda[6]

        self.h_mid = theta*self.h_w + (1-theta)*self.h0_w

        self.P_0 = rho_i*g*H
        self.P_w = rho_w*g*(self.h_w - B)
        self.N = self.P_0 - self.P_w

        self.dt = df.Constant(dt)

        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Nout = df.project(self.N,coupler.Q_cg)
            self.h_file = df.File(results_dir+'h.pvd')
            self.hw_file = df.File(results_dir+'h_w.pvd')
            self.K_file = df.File(results_dir+'K.pvd')
            self.N_file = df.File(results_dir+'N.pvd')
           

    def write_variables(self,t):
        N_temp = df.project(self.N,coupler.Q_cg)
        self.Nout.vector().set_local(N_temp.vector().get_local())
        self.h_file << (self.h0,t)
        self.hw_file << (self.h0_w,t)
        self.N_file << (self.Nout,t)   
        self.K_file << (self.K0,t) 

    def set_boundary_labels(self,edgefunction):
        self.edgefunction = edgefunction
      
    def init_variables(self,hw_init=1e-4,h_init=1e-2,K_init = 1e-2*60**2*24*365):
        self.h0.vector()[:] = h_init
        self.h0_w.vector()[:] = hw_init#= self.coupler.B_c.vector()[:] + 0.8*self.rho_i/self.rho_w*self.coupler.H_c.vector()[:]
        self.K0.vector()[:] = K_init

    def set_timestep(self,dt):
        if self.dt:
            self.dt.assign(dt)
        else:
            self.dt = df.Constant(dt)

    def init_dirichlet_bcs(self):
        self.bcs = [df.DirichletBC(coupler.V.sub(4),df.project(self.coupler.B_c-100),self.edgefunction,1)]

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        La = self.La
        p = self.p
        q = self.q
        beta2 = self.beta2
        omega = self.omega
        G = self.G
        nu = self.nu
        e_v = self.e_v
        ct = self.ct
        cw = self.cw
        alpha = self.alpha
        beta = self.beta
        k = self.k
        h_r = self.h_r 
        l_r = self.l_r
        A = self.A
        theta = self.theta

        normal = df.FacetNormal(mesh)
        edgefunction = self.edgefunction
        ds = df.ds(subdomain_data=edgefunction)

        h = self.h
        h_w = self.h_w
        K = self.K
        h_mid = self.h_mid

        h0 = self.h0
        h0_w = self.h0_w

        xsi = self.xsi
        psi = self.psi
        w = self.w

        dt = self.dt

        m = self.coupler.m

        N = self.N
        Nhat = self.N/(rho_i*g)

        flux = -K*df.grad(h_w)

        u_o = df.sqrt(self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-3)
        O = Max(u_o*(h_r - h)/l_r,0)
        M = 1./La*(G + beta2*Max(N,5e4)**p*u_o**(q-1)*u_o**2 - rho_w*g*df.dot(flux,df.grad(h_w)) - ct*cw*rho_w*df.dot(flux,df.grad(self.P_w)))
        C = A*h*abs(N)**(n-1)*N

        Re = K*((df.dot(df.grad(h_w),df.grad(h_w))+1e-10)**0.5)/(nu)

        eps = 60**2*24*365
        R_K = w*(K - eps**2*abs(h)**3*g/(12*nu*(1+omega*Re)))*df.dx

        R_h = ((h - h0)/dt - O - M/rho_i + C)*psi*df.dx 

        R_hw = (e_v*(h_w - h0_w)/dt*xsi + df.dot(df.grad(xsi),K*df.grad(h_w)) + O*xsi - (1/rho_w - 1./rho_i)*M*xsi - C*xsi - m*xsi)*df.dx# - dt*K*df.dot(df.grad(h_w),normal)*df.ds(2)

        self.coupler.R += R_hw + R_K + R_h

class GLADS(object):
    def __init__(self,mesh):
        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.elements = [E_cg,E_cg,E_cg]

    def set_coupler(self,coupler):
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=214000,p=1,q=1,beta2=2e-3,La=3.35e5,G=0.042*60**2*24*365,k_s=1e-3*60**2*24*365,k_c=1e-1*60**2*24*365,e_v=1e-3,ct=7.5e-8,cw = 4.22e3,alpha=5./4.,beta=3./2.,k=3e5,h_r=0.02,l_r=0.3,A=5.25e-25*60**2*24*365,theta=1.0,dt=1e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)
        self.g = df.Constant(g)
        self.rho_i = df.Constant(rho_i)
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.La = df.Constant(La)
        self.Be = df.Constant(Be)
        self.p = df.Constant(p)
        self.q = df.Constant(q)
        self.beta2 = df.Function(self.coupler.Q_r)
        self.beta2.vector()[:] = beta2
        self.G = df.Constant(G)
        self.e_v = df.Constant(e_v)
        self.ct = df.Constant(ct)
        self.cw = df.Constant(cw)
        self.alpha = df.Constant(alpha)
        self.beta = df.Constant(beta)
        self.k_c = df.Function(self.coupler.Q_r)
        self.k_c.vector()[:] = k_c
        self.k_s = df.Function(self.coupler.Q_r)
        self.k_s.vector()[:] = k_s
        self.h_r = df.Function(self.coupler.Q_r)
        self.h_r.vector()[:] = h_r
        self.l_r = df.Function(self.coupler.Q_r)
        self.l_r.vector()[:] = l_r
        self.A = df.Constant(A)
        self.theta = df.Constant(theta)

        H = self.coupler.H_c
        B = self.coupler.B_c

        self.phi = self.coupler.U[4]
        self.h = self.coupler.U[5]
        self.S = self.coupler.U[6]

        self.phi0 = df.Function(self.coupler.Q_cg)
        self.h0 = df.Function(self.coupler.Q_cg)
        self.S0 = df.Function(self.coupler.Q_cg)

        self.previous_time_vars = [self.phi0,self.h0,self.S0]

        self.xsi = self.coupler.Lambda[4]
        self.psi = self.coupler.Lambda[5]
        self.w = self.coupler.Lambda[6]

        self.P_0 = rho_i*g*H
        phi_m = rho_w*g*B
        self.P_w = self.phi - phi_m
        self.N = self.P_0 - self.P_w

        self.dt = df.Constant(dt)

        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Nout = df.project(self.N,coupler.Q_cg)
            self.phi_file = df.File(results_dir+'phi.pvd')
            self.h_file = df.File(results_dir+'h.pvd')
            self.S_file = df.File(results_dir+'S.pvd')
            self.N_file = df.File(results_dir+'N.pvd')
           
    def write_variables(self,t):
        N_temp = df.project(self.N,coupler.Q_cg)
        self.Nout.vector().set_local(N_temp.vector().get_local())
        self.h_file << (self.h0,t)
        self.phi_file << (self.phi0,t)
        self.N_file << (self.Nout,t)   
        self.S_file << (self.S0,t) 

    def set_boundary_labels(self,edgefunction):
        self.edgefunction = edgefunction
      
    def init_variables(self,phi_init=1e-4,S_init=1e-2,h_init=1e-2):
        self.h0.vector()[:] = h_init
        self.phi0.vector()[:] = self.rho_w*self.g*self.coupler.B_c.vector()[:] + 1000#0.8*self.rho_i/self.rho_w*self.coupler.H_c.vector()[:]
        self.S0.vector()[:] = S_init

    def set_timestep(self,dt):
        if self.dt:
            self.dt.assign(dt)
        else:
            self.dt = df.Constant(dt)

    def init_dirichlet_bcs(self):
        self.bcs = [df.DirichletBC(coupler.V.sub(4),df.project(self.rho_w*self.g*350,self.coupler.Q_cg),self.edgefunction,1)]

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        La = self.La
        p = self.p
        q = self.q
        beta2 = self.beta2
        G = self.G
        e_v = self.e_v
        ct = self.ct
        cw = self.cw
        alpha = self.alpha
        beta = self.beta
        k_c = self.k_c
        k_s = self.k_s
        h_r = self.h_r 
        l_r = self.l_r
        A = self.A
        theta = self.theta

        normal = df.FacetNormal(mesh)
        edgefunction = self.edgefunction
        ds = df.ds(subdomain_data=edgefunction)

        h = self.h
        phi = self.phi
        S = self.S

        h0 = self.h0
        phi0 = self.phi0
        S0 = self.S0

        xsi = self.xsi
        psi = self.psi
        w = self.w

        dt = self.dt

        m = self.coupler.m

        N = self.N

        s = df.as_vector([normal[1],-normal[0]])
        dphids = df.dot(s,df.grad(phi))
        dxsids = df.dot(s,df.grad(xsi))
        Q = -k_c*(S**2+1e-3)**(alpha/2.)*(dphids**2 + 1e-3)**(beta/2. - 1)*dphids
        q = -k_s*(h**2+1e-3)**(alpha/2.)*(df.dot(df.grad(phi),df.grad(phi)) + 1e-3)**(beta/2.-1)*df.grad(phi)

        Chi = -Q*dphids
        Pi = ct*cw*rho_w*Q*dphids

        u_b = df.sqrt(self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-3)
        O = Max(u_b*(h_r - h)/l_r,0)
        C = A*h*abs(N)**(n-1)*N
        C_c = A*S*abs(N)**(n-1)*N

        R_phi = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt - df.dot(df.grad(xsi),q) + xsi*(O - C - m))*df.dx + (-dxsids('+')*Q('+') + xsi('+')*(Chi('+') - Pi('+'))/La*(1./rho_i - 1/rho_w) - xsi('+')*C_c('+'))*df.dS 

        R_h = ((h - h0)/dt - O + C)*psi*df.dx 
        R_S = (((S - S0)/dt - (Chi-Pi)/(rho_i*La) + C_c)*w)('+')*df.dS +  S*w*df.ds

        self.coupler.R += R_phi + R_h + R_S


class ConstantFraction(object):
    def __init__(self,mesh):
        self.elements = []

    def set_coupler(self,coupler):
        self.coupler = coupler

    def set_boundary_labels(self,edgefunction):
        self.edgefunction = edgefunction

    def init_variables(self):
        self.dt
    
    def set_timestep(self,dt):
        if self.dt:
            self.dt.assign(dt)
        else:
            self.dt = df.Constant(dt)

    def build_variables(self,dt=1e-3,f=0.1):
        self.dt = df.Constant(dt)
        self.N = f*self.coupler.H
        self.previous_time_vars = []

    def build_forms(self):
        pass

    def write_variables(self,t):
        pass

class Coupler(object):
    def __init__(self,mesh,stokes,hydro):

        self.stokes = stokes
        self.hydro = hydro

        elements = self.stokes.elements + self.hydro.elements
        E_V = df.MixedElement(elements)  
        self.V = df.FunctionSpace(mesh,E_V)

        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)
     
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.Q_dg = df.FunctionSpace(mesh,E_dg)

        E_r = df.FiniteElement('R',mesh.ufl_cell(),0)
        self.Q_r = df.FunctionSpace(mesh,E_r)
        self.dw = df.TestFunction(self.Q_r)

        self.space_list = [df.FunctionSpace(E,mesh) for E in elements]

        self.U = df.Function(self.V)
        self.Lambda = df.Function(self.V)
        self.Phi = df.TestFunction(self.V)
        self.dU = df.TrialFunction(self.V)

        self.R = 0

    def set_geometry(self,B_c,B_d,H_c,H_d):
        self.B_c = B_c
        self.H_c = H_c
        self.B_d = B_d
        self.H_d = H_d
        self.S = B_c + H_c

    def set_forcing(self,m):
        self.m = m

    def generate_forward_model(self):
        
        function_spaces = [u.function_space() for u in self.stokes.previous_time_vars+self.hydro.previous_time_vars]
        self.assigner_inv = df.FunctionAssigner(function_spaces,self.V)
        self.assigner     = df.FunctionAssigner(self.V,function_spaces)

        self.R_fwd = df.derivative(self.R,self.Lambda,self.Phi)
        self.J_fwd = df.derivative(self.R_fwd,self.U,self.dU)

    def generate_adjoint_model(self,U_obs):
        self.R += ((U_obs[0] - self.stokes.u(0))**2)*df.dx + ((U_obs[1] - self.stokes.v(0))**2)*df.dx
        self.R_adj = df.derivative(self.R,self.U,self.Phi)
        self.J_adj = df.derivative(self.R_adj,self.Lambda,self.dU)
                  
stokes = SpecFO(mesh)
hydro = GLADS(mesh)
#hydro = Shakti(mesh)
#hydro = UnderPressure(mesh)
#hydro = ConstantFraction(mesh)
coupler = Coupler(mesh,stokes,hydro)

B_c = df.Function(coupler.Q_cg,'terrestrial/Bed/B_c.xml')
H_c = df.Function(coupler.Q_cg,'terrestrial/Thk/H_c.xml')
B_d = df.Function(coupler.Q_dg,'terrestrial/Bed/B_d.xml')
H_d = df.Function(coupler.Q_dg,'terrestrial/Thk/H_d.xml')

u_obs = df.Function(coupler.Q_cg,'terrestrial/Vel_x/u_c.xml')
v_obs = df.Function(coupler.Q_cg,'terrestrial/Vel_y/v_c.xml')
U_obs = df.as_vector([u_obs,v_obs])

thklim = 10
Htemp = H_c.vector().get_local()
Htemp[Htemp<thklim] = thklim
H_c.vector().set_local(Htemp)
Htemp = H_d.vector().get_local()
Htemp[Htemp<thklim] = thklim
H_d.vector().set_local(Htemp)

#m = df.Function(coupler.Q_dg,'terrestrial/SMB/smb_d.xml')
m = df.project(0.1+Max((4. - 4./2000*(H_c+B_c)),0))
#m.vector().set_local((np.maximum(-m.vector().get_local()/1000,0)+1e-1))#/(60**2*24*365))
# hydrology FacetFunction, where value is 2 if free-flux, 1 if atmospheric, zero otherwise.
edgefunction = df.MeshFunction('size_t',mesh,1)
for f in df.facets(mesh):
    if f.exterior():
        edgefunction[f] = 2
        if H_c(f.midpoint().x(),f.midpoint().y())<50:
            edgefunction[f]=1

coupler.set_geometry(B_c,B_d,H_c,H_d)
coupler.set_forcing(m)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)

hydro.set_boundary_labels(edgefunction)

stokes.build_variables(results_dir='./glads/')
hydro.build_variables(dt=0.1,results_dir='./glads/')
hydro.init_variables()

stokes.build_forms()
hydro.build_forms()
coupler.generate_forward_model()
coupler.generate_adjoint_model(U_obs)

hydro.init_dirichlet_bcs()

# Nonlinear Problem
problem = df.NonlinearVariationalProblem(coupler.R_fwd,coupler.U,J=coupler.J_fwd,bcs=hydro.bcs)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
solver.parameters['newton_solver']['absolute_tolerance'] = 1e0
solver.parameters['newton_solver']['error_on_nonconvergence'] = True
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['maximum_iterations'] = 12
solver.parameters['newton_solver']['report'] = True

df.File('./glads/U_glads.xml') >> coupler.U
coupler.assigner_inv.assign(stokes.previous_time_vars+hydro.previous_time_vars,coupler.U)

t = 0
t_end = 300
timestep_increase_fraction = 1.2
timestep_reduction_fraction = 0.5
stepsize = 1e-2
dt_max = 0.3
while t<t_end:
    try:
        problem = df.NonlinearVariationalProblem(coupler.R_fwd,coupler.U,J=coupler.J_fwd,bcs=hydro.bcs)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters['nonlinear_solver'] = 'newton'
        solver.parameters['newton_solver']['relaxation_parameter'] = 1.0
        solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
        solver.parameters['newton_solver']['absolute_tolerance'] = 1e0
        solver.parameters['newton_solver']['error_on_nonconvergence'] = True
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        solver.parameters['newton_solver']['maximum_iterations'] = 10
        solver.parameters['newton_solver']['report'] = True
        coupler.assigner.assign(coupler.U,stokes.previous_time_vars+hydro.previous_time_vars)
        coupler.U.vector()[:] += 1e-10*np.random.randn(len(coupler.U.vector().get_local()))
        solver.solve()
        coupler.assigner_inv.assign(stokes.previous_time_vars+hydro.previous_time_vars,coupler.U)

        df.solve(coupler.R_adj==0,coupler.Lambda,J=coupler.J_adj)
        print(df.assemble(((U_obs[0] - stokes.u(0))**2)*df.dx + ((U_obs[1] - stokes.v(0))**2)*df.dx))
        beta2grad = df.assemble(df.derivative(coupler.R,stokes.beta2,coupler.dw))
        hrgrad = df.assemble(df.derivative(coupler.R,hydro.h_r,coupler.dw))
        lrgrad = df.assemble(df.derivative(coupler.R,hydro.l_r,coupler.dw))
        kcgrad = df.assemble(df.derivative(coupler.R,hydro.k_c,coupler.dw))
        ksgrad = df.assemble(df.derivative(coupler.R,hydro.k_s,coupler.dw))

        hydro.beta2.vector()[:] -= stepsize*np.sign(beta2grad.get_local())*hydro.beta2.vector().get_local()
        stokes.beta2.vector()[:] -= stepsize*np.sign(beta2grad.get_local())*stokes.beta2.vector().get_local()
        
        hydro.k_c.vector()[:] -= stepsize*np.sign(kcgrad.get_local())*hydro.k_c.vector().get_local()
        hydro.k_s.vector()[:] -= stepsize*np.sign(ksgrad.get_local())*hydro.k_s.vector().get_local()
        hydro.h_r.vector()[:] -= stepsize*np.sign(hrgrad.get_local())*hydro.h_r.vector().get_local()
        hydro.l_r.vector()[:] -= stepsize*np.sign(lrgrad.get_local())*hydro.l_r.vector().get_local()
        stokes.write_variables(t)
        hydro.write_variables(t)
        t += hydro.dt(0)
        print(hydro.dt(0),t)
        hydro.dt.assign(min(hydro.dt(0)*timestep_increase_fraction,dt_max))
    except RuntimeError:
        hydro.dt.assign(hydro.dt(0)*timestep_reduction_fraction)
        print('Convergence not achieved.  Reducting time step to {0} and trying again'.format(hydro.dt(0)))

    

        
        
