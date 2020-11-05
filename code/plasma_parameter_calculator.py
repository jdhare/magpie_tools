import numpy as np
import locale
locale.setlocale(locale.LC_ALL, '') #this lets us put commas every three digits
import scipy.constants as cons


#useful function tht really should be built in....rounds to n sig figs
round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1)) 

def str_nsf(x, n):
    #returns a string to n significant figures, and nicely formatted too!
    x_sig_figs=int(np.floor(np.log10(np.abs(x)))) #sig figs of x
    xsf=round_to_n(x,n)
    if xsf>=1:
        p=locale.format("%.0f",xsf, grouping=True)
    else:
        p=locale.format("%."+str(np.abs(x_sig_figs)+n-1)+"f",xsf, grouping=True)
    return p

class Plasma:
    '''
    Usage eg:
        Note: n_e in cm^-3, T_e, T_i in eV, B in Tesla, V in m/s, L in mm
        al_flow={'A':27,'Z':3.5,'n_e':5e17,'T_e': 25,'T_i':25,'B': 2, 'V':50e3,'L':7.25}
        al=Plasma(**al_flow)
        print(al.beta_dyn) #prints specific parameter
        al.print_dimensionless_parameters() # rpints selected highlights
    '''
    def __init__(self, A, Z, n_e, T_e, T_i, B, V, L):
        '''
        Args:
            A: ion mass in nucleon masses
            Z: average charge state
            n_e: electron density in cm^-3
            T_e electron temperature in eV
            T_i ion temperature in eV
            B: magnetic field in Tesla
            V: velocity in m/s
            L: length scale in mm.
        '''
        self.T_e=T_e*cons.e/cons.k
        self.T_i=T_i*cons.e/cons.k
        self.A=A
        self.Z=Z
        self.n_e=n_e*1e6
        self.B=B
        self.V=V
        self.L=L*1e-3
        self.params()
    def params(self):
        B=np.abs(self.B)
        Z=self.Z
        A=self.A
        m_i=A*cons.m_u
        n_i=self.n_e/Z
        n_e=self.n_e
        T_i=self.T_i
        T_e=self.T_e
        V=np.abs(self.V)
        
        e=cons.e
        mu_0=cons.mu_0
        c=cons.c
        m_e=cons.m_e
        k=cons.k
        
        self.calculate_columb_logarithms()
        
        # Velocities
        self.V_te=np.sqrt(k*T_e/m_e)
        self.V_ti=np.sqrt(k*T_i/m_i)
        self.V_S=np.sqrt(k*(Z*T_e+T_i)/m_i)
        self.V_A=np.sqrt(B**2/(mu_0*n_i*m_i))

        # Frequencies
        self.om_ce=e*B/m_e
        self.om_ci=Z*e*B/m_i
        self.om_pe=np.sqrt(e**2*n_e/(cons.epsilon_0*m_e))
        self.om_pi=np.sqrt(Z**2*e**2*n_i/(cons.epsilon_0*m_i))
        self.calculate_collision_frequencies()
        
        # Time scales:
        self.tau_ei=1/self.nu_ei
        self.tau_E_ei=self.tau_ei*A*(cons.m_u/(2*m_e))
        try:
            self.tau_ie_cooling=T_i/(Z*(T_e-T_i))*self.tau_E_ei
        except ZeroDivisionError:
            pass
            
        #resistvity
        self.sigma_0=n_e*e**2/(m_e*self.nu_ei) #s kg^-3 m^-3 C^-3
        self.D_M=1/(mu_0*self.sigma_0) #m^2/s, eta/mu_0
        self.eta=self.D_M*mu_0
        #self.eta=(np.pi*Z*e**2*m_e**0.5*self.col_log_ei)/((4*np.pi*cons.epsilon_0)**2*(k*T_e)**1.5) #Chen PPCF

        #length scales
        self.la_de=np.sqrt(cons.epsilon_0*k*T_e/(n_e*e**2))
        self.delta_i=c/self.om_pi#ion skin depth
        self.delta_e=c/self.om_pe#electron skin depth
        self.rho_i=self.V_ti/self.om_ci
        self.rho_e=self.V_te/self.om_ce
        self.mfp_i=self.V_ti/self.nu_ie#cm
        self.mfp_e=self.V_te/self.nu_ei#cm
        self.l_visc=(self.visc/self.V) # Re=1, viscous time scale= advective time scale
        
        #pressures
        self.P_B=B**2/(2*mu_0)
        self.P_th=n_i*k*(Z*T_e+T_i)
        self.P_ram=n_i*m_i*V**2
        
        #energy densities:
        self.E_B=self.P_B
        self.E_th=self.P_th
        self.E_kin=self.P_ram/2
        
        #dimensionless parameters
        self.beta_th=self.P_th/self.P_B
        self.beta_dyn=self.P_ram/self.P_B
        self.M_S=V/self.V_S
        self.M_A=V/self.V_A
        self.i_mag=self.om_ci/self.nu_ie
        self.e_mag=self.om_ce/self.nu_ei
        self.Re_M=mu_0*self.L*V/self.eta
        self.S=mu_0*self.L*self.V_A/self.eta

        
        #dimensionless parameters dodgy?
        self.Re=self.L*self.V/self.visc
        self.Pr_M=self.Re_M/self.Re
          
    def calculate_columb_logarithms(self):
        '''
        Switches back to CGS units to use the convienient formulas from the NRL formulary,
        to calculate the three Coloumb logarithms.
        '''
        T_e=self.T_e*cons.k/cons.e
        T_i=self.T_i*cons.k/cons.e
        n_e=self.n_e*1e-6
        Z=self.Z
        n_i=n_e/Z
        #Lots of definitions of the coloumb logarithm
        self.col_log_ee=23.5-np.log(n_e**0.5*T_e**-1.25)-(1e-5+((np.log(T_e-2))**2)/16.0)**0.5
        self.col_log_ii=23-np.log(np.sqrt(2)*n_i**0.5*Z**3*T_i**-1.5)
        if T_e<10*Z**2: #see NRL formulary pg 34
            self.col_log_ei=23-np.log(n_e**0.5*Z*T_e**-1.5)
        else:
            self.col_log_ei=24-np.log(n_e**0.5*T_e**-1.0)
            
        self.visc=2e19*T_i**2.5/(self.col_log_ei*self.A**0.5*Z**4*n_i) #Ryutov 99 cm^2/s
        self.visc=self.visc*1e-4 #convert to m^2/s

    def calculate_collision_frequencies(self):
        '''
        Switches back to CGS units to use the convienient formulas from the NRL formulary,
        to calculate the three collision frequencies
        '''        
        T_e=self.T_e*cons.k/cons.e
        T_i=self.T_i*cons.k/cons.e
        n_e=self.n_e*1e-6
        Z=self.Z
        n_i=n_e/Z
        A=self.A

        self.nu_ei=2.91e-6*Z*n_e*self.col_log_ei*T_e**-1.5#electrons on ions
        self.nu_ie=4.80e-8*Z**4*A**-0.5*n_i*self.col_log_ei*T_i**-1.5 #ions on electrons
        self.nu_eq=1.8e-19*(A*cons.m_u*cons.m_e)**0.5*Z**2*n_e*self.col_log_ei/(A*cons.m_u*T_e+cons.m_e*T_i)**1.5
        
        self.ion=Particle(m=A, Z=Z, T=T_i, v=None,n=n_i)
        self.electron=Particle(m=cons.m_e/cons.m_u, Z=-1, T=T_e, v=None,n=n_e)
                

    def print_dim_params(self):
        im='Ion magnetisation = '+str(round_to_n(self.i_mag,2))
        em='Electron magnetisation = '+str(round_to_n(self.e_mag,2))
        b='Thermal Beta = '+str(round_to_n(self.beta_th,2))
        br='Dynamic Beta = '+str(round_to_n(self.beta_dyn,2))
        rm='Magnetic Reynolds Number = '+str(round_to_n(self.Re_M,2))
        S='Lundquist number = '+str(round_to_n(self.S,2))
        m='Mach number = '+str(round_to_n(self.M_S,2))
        ma='Mach-Alfven number = '+str(round_to_n(self.M_A,2))

        txtstr=im+'\n'+em+'\n'+b+'\n'+br+'\n'+m+'\n'+ma+'\n'+rm+'\n'+S
        print(txtstr)

def col_log_eis(T_e,n_e,Z):
    if T_e<10*Z**2: #see NRL formulary pg 34
        col_log=23-np.log(n_e**0.5*Z*T_e**-1.5)
    else:
        col_log=24-np.log(n_e**0.5*T_e**-1.0)
    return col_log
col_log_ei=np.vectorize(col_log_eis)

#CGS for v, n, eV for T, m in atomic mass units
#test particle a slows on a field of particles b
#nu_ab is not equal to nu_ba!!!
c=cons
class Particle:
    def __init__(self, m, Z, v, T, n):
        self.m=float(m)
        self.m_g=m*c.m_u*1e3 #convert to grams
        self.T_erg=T*c.e*1e7#convert to ergs
        self.T=float(T)
        self.Z=Z
        self.v_T=np.sqrt(self.T_erg/self.m_g)
        if v==None:
            self.v=self.v_T
        else:
            self.v=float(v)
        self.n=float(n)
        self.e=Z*4.8e-10#in stat coloumbs
        
def x_ab(a,b):
    return b.m_g*a.v**2/(2*b.T_erg)
    
def col_log(a,b, T_e=None):
    if a.Z is -1 and b.Z is -1:#electron electron
        print('Electron-Electron')
        return 23.5-np.log(a.n**0.5*a.T**-1.25)-(1e-5+((np.log(a.T-2))**2)/16.0)**0.5
    if a.Z is not -1 and b.Z is not -1: #ion ion
        v_D=np.abs(a.v-b.v)       
        if a.v_T<v_D and b.v_T<v_D:
            print('Counter-streaming ions')
            beta_D=v_D/(c.c*1e2)
            n_e=a.Z*a.n+b.Z*b.n
            return 43-np.log(a.Z*b.Z*(a.m+b.m)/(a.m*b.m*beta_D**2)*(n_e/T_e)**0.5)
        else:
            print('Mixed ion-ion')
            return 23-np.log(a.Z*b.Z*(a.m+b.m)/(a.m*a.T+b.m*b.T)*(a.n*a.Z**2/a.T+b.n*b.Z**2/b.T)**0.5)
    else: #electron ion
        if a.Z is -1:
            el=a
            io=b
            print('b is ion, a is electron')
        else:
            el=b
            io=a
            print('a is ion, b is electron')
        #Define the three 'decision temperatures'
        Te=el.T
        Ti=io.T*el.m/io.m
        TZ=10*io.Z**2
        if Ti<Te and Te<TZ: #see NRL formulary pg 34
            print('Ion-Electron, T_i*m_e/m_i<T_e<10 Z^2')
            return 23-np.log(el.n**0.5*io.Z*el.T**-1.5)
        elif Ti<TZ and TZ<Te:
            print('Ion-Electron, T_i*m_e/m_i<10 Z^2<T_e')
            return 24-np.log(el.n**0.5*el.T**-1.0)
        elif Te<io.Z*Ti:
            print('SUSPECT! Ion-Electron, T_i*m_e/m_i<10 Z^2<T_e')
            return 30-np.log(io.n**0.5*io.T**-1.5*io.Z**2/io.m)
        else:
            print('Ion-Electron: Whoops! You broke Physics!!!')
            return 2

        
def nu_0(a,b, T_e=None):
    return 4*np.pi*a.e**2*b.e**2*col_log(a,b, T_e)*b.n/(a.m_g**2*np.abs(a.v)**3)

def psi(x, steps=1e6):
    t=np.linspace(0, x, int(steps))
    integrand=t**0.5*np.exp(-t)
    return 2/np.sqrt(np.pi)*np.trapz(integrand, x=t)

def psi_prime(x):
    return 2/np.sqrt(np.pi)*x**0.5*np.exp(-x)

def nu_slowing(a,b, T_e=None):
    xab=x_ab(a,b)
    pab=psi(xab)
    return (1+a.m/b.m)*pab*nu_0(a,b, T_e)

def nu_transverse(a,b, T_e=None):
    xab=x_ab(a,b)
    pab=psi(xab)
    ppab=psi_prime(xab)
    return 2*((1-1/(2*xab))*pab+ppab)*nu_0(a,b, T_e)

def nu_parallel(a,b, T_e=None):
    xab=x_ab(a,b)
    pab=psi(xab)
    return (pab/xab)*nu_0(a,b, T_e)

def nu_energy(a,b, T_e=None):
    xab=x_ab(a,b)
    pab=psi(xab)
    ppab=psi_prime(xab)
    return 2*((a.m/b.m)*pab-ppab)*nu_0(a,b, T_e)

def l_slowing(a,b, T_e=None):
    return a.v/nu_slowing(a,b,T_e)

def l_parallel(a,b, T_e=None):
    return a.v/nu_parallel(a,b,T_e)

def l_transverse(a,b, T_e=None):
    return a.v/nu_transverse(a,b,T_e)

def l_energy(a,b, T_e=None):
    return a.v/nu_energy(a,b,T_e)
    