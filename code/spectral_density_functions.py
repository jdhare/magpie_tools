import numpy as np
from numpy import sqrt
import scipy.constants
import scipy.special

m_e=scipy.constants.m_e
m_p=scipy.constants.m_p
e=scipy.constants.e
c=scipy.constants.c
epsilon_0=scipy.constants.epsilon_0

exp=np.exp
Il=scipy.special.iv #modified bessel function of first kind

#round to n significant figures, good for stringifying.
round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))

'''def convolve(arr, kernel):
    # simple convolution of two arrays
    # can't remember where I took this from or how it works, but it does.
    npts = min(len(arr), len(kernel))
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts)/2)
    return out[noff:noff+npts]'''

'''def convolve(arr, kernel):
    forward = np.convolve(arr, kernel, mode='same')
    backward = np.convolve(arr[::-1], kernel, mode='same')[::-1]
    return (forward+backward)/2'''

def convolve(arr,kernel):
    return np.convolve(arr,kernel, mode='same')


def S_k_omega(lambda_range, lambda_in, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0):
    '''
    Returns a normalised spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temeperatures
    Electron velocity is with respect to ion velocity
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in metres
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    '''

    #physical parameters
    pi=np.pi
    Z=Z_nLTE(T_e, Z_Te_table)
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_e=(omega/k+v_fe+v_fi)/a
    x_i=(omega/k+v_fi)/b
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) #ion susceptibility
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))#electron susceptibility
    epsilon=1+chi_e+chi_i#dielectric function
    fe0=1/(sqrt(pi)*a)*np.exp(-x_e**2)#electron Maxwellian function
    fi0=1/(sqrt(pi)*b)*np.exp(-x_i**2)#ion Maxwellian
    Skw=2*pi/k*(abs(1-chi_e/epsilon)**2*fe0+Z*abs(chi_e/epsilon)**2*fi0)
    return Skw/Skw.max() #normalise the spectrum

def S_k_omega_conv(lambda_range, lambda_in, response, theta, A, T_e,T_i,n_e,Z, v_fi=0, v_fe=0, amp=1.0):
    '''
    Takes the Spectral density function and convolves it with a given Instrument Response.
    It's necessary to have an extra function just to do this for lmfit.
    Otherwise it would be ridiculous.
    '''
    skw=S_k_omega(lambda_range, lambda_in, theta, A, T_e,T_i,n_e,Z, v_fi, v_fe)
    skw_conv=convolve(response,skw)
    return amp*skw_conv/skw_conv.max()

def S_k_omega_e(lambda_range, lambda_in, theta,T_e,n_e, v_fe):
    '''
    Returns a normalised spectral density function for the electron component only
    Implements the model of Sheffield (2nd Ed.)
    One electron species
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in metres
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e in eV, n_e in cm^-3
    V_fe in m/s
    '''
    #physical parameters
    pi=np.pi
    om_pe=5.64e4*n_e**0.5#electron plasma frequency
    #define omega and k as in Sheffield 113

    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi#convert to radians for cosine function
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    x_e=(omega/k+v_fe)/a
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz

    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0)

    Skw *= lambda_range**-2 # correct for distortion due to the jacobian. Small effect so leave it out

    return Skw/Skw.max() #normalise the spectrum


def Skw_e_stray_light_convolve(lambda_range, interpolation_scale, lambda_in, response, theta, n_e, T_e, V_fe, stray, amplitude, offset, shift, notch):
    skw=S_k_omega_e(lambda_range, lambda_in, theta, T_e, n_e, V_fe)
    skw_conv=convolve(response,skw)
    skw_conv_stray=amplitude*skw_conv/skw_conv.max()+stray*response/response.max()+offset #add in some of the background to account for unshifted light
    return skw_conv_stray[::interpolation_scale]*notch

def S_k_omega_e_resonance(lambda_range, lambda_in, theta,T_e,n_e, v_fe):
    '''
    Returns a normalised spectral density function for the electron component only
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temeperatures
    Electron velocity is with respect to ion velocity
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in metres
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    '''
    #physical parameters
    pi=np.pi
    om_pe=5.64e4*n_e**0.5#electron plasma frequency
    #define omega and k as in Sheffield 113
    ki=2*pi/lambda_in
    omega_i=((c*ki)**2+om_pe**2)**0.5

    ks=2*pi/lambda_range
    omega_s=((c*ks)**2+om_pe**2)**0.5

    th=theta/180.0*np.pi#convert to radians for cosine function
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i

    res= np.sqrt(omega_p**2)
    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    x_e=(omega/k+v_fe)/a
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    omega_p=np.sqrt(n_e*e**2/(m_e*epsilon_0))
    res=np.sqrt(omega_p**2*(1+3/alpha**2))

    #set up the Fadeeva function
    w=scipy.special.wofz

    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0)

    return Skw/Skw.max() #normalise the spectrum

def Skw_e_res_stray_light_convolve(lambda_range, interpolation_scale, lambda_in, response, theta, n_e, T_e, V_fe, stray, amplitude, offset, shift, notch):
    skw=S_k_omega_e(lambda_range, lambda_in, theta, T_e, n_e, V_fe)
    skw_conv=convolve(response,skw)
    skw_conv_stray=amplitude*skw_conv/skw_conv.max()+stray*response/response.max()+offset #add in some of the background to account for unshifted light
    return skw_conv_stray[::interpolation_scale]*notch

def S_k_omega_V3D(lambda_range, lambda_in, theta_0, theta, phi, A, T_e,T_i,n_e,Z, V_fi, theta_V, phi_V):
    '''
    Spectral density function for a plasma with a velocity vector
    Returns a normalised spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temperatures
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in metres
    Theta is angle between k_in and k_s in degrees
    Phi is the angle below the plane formed by k_in and the centre of the aperture in degrees
    theta_V is the polar angle of the velocity (angle wrt to the centre of the lens)
    phi_V is the azimuthal angle of the velocity (angle wrt scattering plane) See J Hare thesis fig 2.13
    A is atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    V_fe must be 0
    '''
    #physical parameters
    pi=np.pi
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5#electron plasma frequency
    #define omega and k as in Sheffield 113
    ki=2*pi/lambda_in
    omega_i=((c*ki)**2+om_pe**2)**0.5
    th0=theta_0/180.0*np.pi#convert to radians for cos, sin funcs.
    ki_vec=np.array([np.sin(th0),0.0,np.cos(th0)])*ki#vector ki in direction [0,1,0]

    ks=2*pi/lambda_range
    omega_s=((c*ks)**2+om_pe**2)**0.5
    th=theta/180.0*np.pi#convert to radians for cos, sin funcs.
    ph=phi/180.0*np.pi
    ks_vec=np.array([[ksi*np.sin(th)*np.cos(ph),ksi*np.sin(th)*np.sin(ph), ksi*np.cos(th)] for ksi in ks])

    k_vec=ks_vec-ki_vec#vector k given by difference of two
    k=np.array([np.sqrt(kvi[0]**2+kvi[1]**2+kvi[2]**2) for kvi in k_vec])#magnitude, (k_x**2+k_y**+k_z**2)**0.5
    omega=omega_s-omega_i#corresponding frequency shift

    th_V=theta_V/180.0*np.pi#phi_B defined wrt to k_i in plane
    ph_V=phi_V/180.0*np.pi#psi_B defined wrt to k_i out of plane

    V=V_fi*np.array([np.sin(th_V)*np.cos(ph_V), np.sin(th_V)*np.sin(ph_V), np.cos(th_V)])#unit vector in direction of B

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    kdotV=np.array([np.vdot(kk, V) for kk in k_vec])
    x_i=(omega+kdotV)/(k*b)
    x_e=(omega+kdotV)/(k*a)
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) #ion susceptibility
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))#electron susceptibility
    fe0=np.exp(-x_e**2)/a
    fi0=np.exp(-x_i**2)/b
    ions=Z*fi0
    epsilon=1+chi_e+chi_i

    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0+np.abs(chi_e/epsilon)**2*ions)
    return Skw/Skw.max() #normalise the spectrum

def exp_Il_approx(l,z):
    '''Asymptotic expansion of Il for large z allows us to cancel two huge exponential pre-factors'''
    mu=4*float(l)**2
    return 1/sqrt(2*np.pi*z)*(1-(mu-1)/(8*z)+(mu-1)*(mu-9)/(2*(8*z)**2)-(mu-1)*(mu-9)*(mu-25)/(6*(8*z)**3))

def S_k_omega_emag3D(lambda_range, lambda_in, theta_0, theta, phi, A, T_e,T_i,n_e,Z, B, theta_B, phi_B, n_har=10, v_fi=0, v_fe=0):
    '''
    Spectral density function fo a plasma with electrons but not ions magnetised
    Full 3D vector implementation to account for the anisotropy of a Maxwellian in a magnetic field.
    Returns a normalised spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temperatures
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in metres
    Theta is angle between k_in and k_s in degrees
    Phi is the angle below the plane formed by k_in and the centre of the aperture in degrees
    Confusing, phi_B is the angle the magnetic field makes with k_i in the plane
    And psi_B is the angle the magnetic field makes with the plane
    B is in Tesla
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    Does not reduce to unmagnetised function for low B due to the approximations use
    Does not properly implement V_fi and V_fe. These should always be 0.
    '''
    #physical parameters
    pi=np.pi
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5#electron plasma frequency
    #define omega and k as in Sheffield 113
    ki=2*pi/lambda_in
    omega_i=((c*ki)**2+om_pe**2)**0.5
    th0=theta_0/180.0*np.pi#convert to radians for cos, sin funcs.
    ki_vec=np.array([np.sin(th0),0.0,np.cos(th0)])*ki#vector ki in direction [0,1,0]

    ks=2*pi/lambda_range
    omega_s=((c*ks)**2+om_pe**2)**0.5
    th=theta/180.0*np.pi#convert to radians for cos, sin funcs.
    ph=phi/180.0*np.pi
    ks_vec=np.array([[ksi*np.sin(th)*np.cos(ph),ksi*np.sin(th)*np.sin(ph), ksi*np.cos(th)] for ksi in ks])

    k_vec=ks_vec-ki_vec#vector k given by difference of two
    k=np.array([np.sqrt(kvi[0]**2+kvi[1]**2+kvi[2]**2) for kvi in k_vec])#magnitude, (k_x**2+k_y**+k_z**2)**0.5
    omega=omega_s-omega_i#corresponding frequency shift

    th_B=theta_B/180.0*np.pi#phi_B defined wrt to k_i in plane
    ph_B=phi_B/180.0*np.pi#psi_B defined wrt to k_i out of plane

    b_hat_par=np.array([np.sin(th_B)*np.cos(ph_B), np.sin(th_B)*np.sin(ph_B), np.cos(th_B)])#unit vector in direction of B

    k_par=np.array([np.abs(np.vdot(kk, b_hat_par)) for kk in k_vec]) #component of k paralle to field. Absolute value taken
    k_per=np.array([np.sqrt(kt**2-kp**2) for kp, kt in zip(k_par,k)])#component of k perpendicular to field. Positive sqrt taken
    Omega_e=e*B/m_e#electron cyclotron frequency

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_i=(omega/k+v_fi)/b
    x_eC=k_per*a/(sqrt(2)*Omega_e)#k_perpendicular*thermal gyro-radius

    lambda_De=7.43*(T_e/n_e)**0.5 #in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) #ion susceptibility
    He_terms=[]
    for l in np.arange(-n_har,n_har+1):
        x_el=((omega-l*Omega_e)/k_par)/a
        Il_x_eC=Il(l,x_eC**2)
        if np.isinf(Il_x_eC).any() or np.isnan(Il_x_eC).any():
            exp_Il=exp_Il_approx(l,x_eC**2)
        else:
            exp_Il=Il_x_eC*exp(-x_eC**2)
        He_l=exp_Il*omega/(omega-l*Omega_e)*1j*sqrt(pi)*x_el*np.conj(w(x_el))
        He_terms.append(He_l)
    H_e=alpha**2*(1+sum(He_terms))#sum over all the calculated harmonics to get electron susceptibility
    epsilon=1+H_e+chi_i #dielectric function
    fe0_terms=[]
    for l in np.arange(-n_har,n_har+1):
        x_el=((omega-l*Omega_e)/k_par)/a
        Il_x_eC=Il(l,x_eC**2)
        if np.isinf(Il_x_eC).any() or np.isnan(Il_x_eC).any():
            exp_Il=exp_Il_approx(l,x_eC**2)
        else:
            exp_Il=Il_x_eC*exp(-x_eC**2)
        fe0_l=exp_Il*np.exp(-x_el**2)
        fe0_terms.append(fe0_l)
    fe0=sum(fe0_terms)/a #sum over all harmonics to get electron distribution
    fi0=np.exp(-x_i**2)/b #ion distribution
    Skw_ef=2*sqrt(pi)*np.abs(1-H_e/epsilon)**2*(fe0/k_par)
    Skw_if=2*sqrt(pi)*Z*np.abs(H_e/epsilon)**2*(fi0/k)#eq 10.3.11 Sheffield
    Skw=Skw_ef+Skw_if
    return Skw/Skw.max() #normalise the spectrum


def aperture_integ_and_convolve(func, aperture_angle, steps, response):
    '''
    Integrates a given function over sin(theta) dtheta dphi over 2pi in phi
    and from 0 to aperture_angle in theta. func must accept only two arguemnts,
    theta and phi, so a lambda function is helpful to fix other parameters
    Afterwards, convolves with a given response function.
    '''
    res=[]
    aa=aperture_angle
    for t in np.linspace(aa/steps, aa, steps): #don't bother evaluating t=0, as sin(0)=0
        for p in np.linspace(0, 360.0, steps):
            res.append(np.sin(t)*func(t,p))
    res=np.sum(res,0)
    res_conv=convolve(response,res)
    res_conv_norm=res_conv/res_conv.max()
    return res_conv_norm

def aperture_angle(distance, diameter):
    '''Calculates the half opening angle in degrees of an aperture of a given diameter at a given distance'''
    return diameter/(2.0*distance)*180.0/np.pi

def k_dot_B(theta_0, theta_B, phi_B):
    th0=theta_0/180.0*np.pi#convert to radians for cos, sin funcs.
    ki_vec=np.array([np.sin(th0),0.0,np.cos(th0)])
    ks_vec=np.array([0,0,1])
    k_vec=ks_vec-ki_vec
    thB=theta_B*np.pi/180.0
    phB=phi_B*np.pi/180.0
    b_hat_par=np.array([np.sin(thB)*np.cos(phB), np.sin(thB)*np.sin(phB), np.cos(thB)])#unit vector in direction of B
    return np.vdot(k_vec, b_hat_par)

def str_to_n(x,n):
    if x==0:
        return '0'
    sf=int(np.floor(np.log10(np.abs(x))))
    r=round_to_n(x,n)
    if sf>=n-1:
        r=int(r)
    return str(r)

def S_k_omega_mspecies(lambda_range, lambda_in, theta,  T_e,T_i,n_e, Aj,Zj,Fj, V_fi, V_fe):
    #physical parameters
    pi=np.pi
    m_ij=[m_p*Aj for Aj in Aj]
    ZF_sum=sum([Z*F for Z, F in zip(Zj,Fj)])
    scale=n_e/ZF_sum
    Nj=[scale*F for F in Fj]
    N_tot=sum([Z*N for Z,N in zip(Zj,Nj)])
    om_pe=5.64e4*n_e**0.5
    #define omega and k as in Sheffield 113
    ki=2*pi/lambda_in
    omega_i=((c*ki)**2+om_pe**2)**0.5
    ks=2*pi/lambda_range
    omega_s=((c*ks)**2+om_pe**2)**0.5
    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i
    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    bj=[sqrt(2*e*T_i/m_i) for m_i in m_ij]
    x_e=(omega/k+V_fe+V_fi)/a
    x_ij=[(omega/k+V_fi)/b for b in bj]
    lambda_De=7.43*(T_e/n_e)**0.5 #in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    #susceptibilities
    chi_ij=[alpha**2*Z**2*N/N_tot*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) for Z, N, x_i in zip(Zj,Nj,x_ij)]
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e+sum(chi_ij)
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    fi0j=[np.exp(-x_i**2)/b for b, x_i in zip(bj,x_ij)]
    ions=sum([Z**2*N/N_tot*fi0 for Z,N,fi0 in zip(Zj,Nj,fi0j)])
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0+np.abs(chi_e/epsilon)**2*ions)
    return Skw/Skw.max() #normalise the spectrum

def S_k_omega_two_stream(lambda_range, lambda_in, theta,n_e, T_e, V_fe, Aj, Zj, Fj, T_i1, V_fi1, T_i2, V_fi2):
    v_fij=[V_fi1, V_fi2]
    T_ij=[T_i1, T_i2]
    v_fi_bar=np.array([F*v_fi for F, v_fi in zip(Fj, v_fij)]).sum()/Fj.sum() #average ion velocity
    #physical parameters
    pi=np.pi
    m_ij=[m_p*Aj for Aj in Aj]
    ZF_sum=sum([Z*F for Z, F in zip(Zj,Fj)])
    scale=n_e/ZF_sum
    Nj=[scale*F for F in Fj]
    N_tot=sum([Z*N for Z,N in zip(Zj,Nj)])
    om_pe=5.64e4*n_e**0.5
    #define omega and k as in Sheffield 113
    ki=2*pi/lambda_in
    omega_i=((c*ki)**2+om_pe**2)**0.5
    ks=2*pi/lambda_range
    omega_s=((c*ks)**2+om_pe**2)**0.5
    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i
    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    bj=[sqrt(2*e*T_i/m_i) for m_i, T_i in zip(m_ij, T_ij)]
    x_e=(omega/k+V_fe+v_fi_bar)/a
    x_ij=[(omega/k+v_fi)/b for b, v_fi in zip(bj, v_fij)]
    lambda_De=7.43*(T_e/n_e)**0.5 #in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    #susceptibilities
    chi_ij=[alpha**2*Z**2*N/N_tot*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) for Z, N, x_i, T_i in zip(Zj,Nj,x_ij, T_ij)]
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e+sum(chi_ij)
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    fi0j=[np.exp(-x_i**2)/b for b, x_i in zip(bj,x_ij)]
    ions=sum([Z**2*N/N_tot*fi0 for Z,N,fi0 in zip(Zj,Nj,fi0j)])
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0+np.abs(chi_e/epsilon)**2*ions)
    return Skw/Skw.max() #normalise the spectrum

def Skw_multi_species_stray_light_convolve(lambda_range, lambda_in, response, theta,  n_e, T_e, V_fe, Aj, Zj, Fj, T_i1, V_fi1, stry, amp, offset):
    skw=S_k_omega_mspecies(lambda_range, lambda_in, theta,  T_e,T_i1,n_e, Aj,Zj,Fj, V_fi1, V_fe)
    skw_conv=convolve(response,skw)
    skw_conv_norm=skw_conv/skw_conv.max()
    skw_conv_norm_stray=amp*skw_conv_norm+stry*response+offset #add in some of the background to account for unshifted light
    return skw_conv_norm_stray

def Skw_two_stream_stray_light_convolve(lambda_range, lambda_in, response, theta,  n_e, T_e, V_fe, Aj, Zj, Fj, T_i1, V_fi1, T_i2, V_fi2, stry, amp, offset):
    skw=S_k_omega_two_stream(lambda_range, lambda_in, theta, n_e, T_e, V_fe, Aj, Zj, Fj, T_i1, V_fi1, T_i2, V_fi2)
    skw_conv=convolve(response,skw)
    skw_conv_norm=skw_conv/skw_conv.max()
    skw_conv_norm_stray=amp*skw_conv_norm+stry*response+offset #add in some of the background to account for unshifted light
    return skw_conv_norm_stray

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def Z_nLTE(Te, Z_Te_table):
    Te_list=Z_Te_table[:,0]
    index=find_nearest(Te_list, Te)
    Z=Z_Te_table[index,1]
    return Z

def S_k_omega_nLTE_abs(lambda_range, lambda_in, response, theta,  Z_Te_table, T_e, T_i, n_e, A,V_fi, V_fe):
    #physical parameters
    pi=np.pi
    Z=Z_nLTE(T_e, Z_Te_table)
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift
    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_e=(omega/k+V_fe+V_fi)/a
    x_i=(omega/k+V_fi)/b
    lambda_De=7.43*(T_e/n_e)**0.5 #in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    #susceptibilities
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i))
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e+chi_i
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    fi0=np.exp(-x_i**2)/b
    ions=Z*fi0
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0+np.abs(chi_e/epsilon)**2*ions)

    skw_conv=convolve(response,skw)

    return Skw #don't normalise the spectrum

def Skw_nLTE_stray_light_convolve_abs(lambda_range, interpolation_scale, lambda_in, response, theta,  Z_Te_table, n_e, T_e, V_fe, A, T_i, V_fi, stray, amplitude, offset, shift, notch):
    skw=S_k_omega_nLTE(lambda_range, lambda_in, theta,  Z_Te_table, T_e, T_i, n_e, A,V_fi, V_fe)
    skw_conv=convolve(response,skw)
    skw_conv_stray=amplitude*skw_conv #add in some of the background to account for unshifted light
    return skw_conv_stray[::interpolation_scale]*notch



def S_k_omega_nLTE(lambda_range, lambda_in, theta,  Z_Te_table, T_e, T_i, n_e, A,V_fi, V_fe):
    #physical parameters
    pi=np.pi
    #Z=Z_nLTE(T_e, Z_Te_table)
    Z = Z_Te_table(T_e)
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift
    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_e=(omega/k+V_fe+V_fi)/a
    x_i=(omega/k+V_fi)/b
    lambda_De=7.43*(T_e/n_e)**0.5 #in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    #susceptibilities
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i))
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))
    epsilon=1+chi_e+chi_i
    #distribution functions
    fe0=np.exp(-x_e**2)/a
    fi0=np.exp(-x_i**2)/b
    ions=Z*fi0
    Skw=2*sqrt(pi)/k*(np.abs(1-chi_e/epsilon)**2*fe0+np.abs(chi_e/epsilon)**2*ions)
    if np.isnan(Skw.sum()):
        print("Nan alert:",T_e,T_i,n_e,V_fi,V_fe)
    return Skw/Skw.max() #normalise the spectrum


def Skw_nLTE_stray_light_convolve(lambda_range, interpolation_scale, lambda_in, response, theta,  Z_Te_table, n_e, T_e, V_fe, A, T_i, V_fi, stray, amplitude, offset, shift, notch):
    skw=S_k_omega_nLTE(lambda_range, lambda_in, theta,  Z_Te_table, T_e, T_i, n_e, A,V_fi, V_fe)
    skw_conv=convolve(response,skw)
    skw_conv_stray=amplitude*skw_conv/skw_conv.max()+stray*response/response.max()+offset #add in some of the background to account for unshifted light
    return skw_conv_stray[::interpolation_scale]*notch


