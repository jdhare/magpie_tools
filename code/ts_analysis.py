from spectral_density_functions import *
import csv
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import VoigtModel
import os
import numpy as np
from scipy.interpolate import CubicSpline


class Fibre:
    def __init__(self, wavelength, background, shot, bkgd_err, shot_err, theta):
        self.lamb=wavelength*1e-9
        self.bkgd=background
        self.shot=shot
        self.shot_err=shot_err
        self.bkgd_err=bkgd_err
        self.theta=theta
        self.params={}
    def voigt_response(self, sigma=None, gamma=None, weights=True):
        '''
        Fit the background with a Voigt profile to determine the response
        of the spectrometer

        If you have a good, clear signal, set sigma and gamma to None (done by default)

        If your signal is poor, set sigma and gamma using a fit to a good signal, and then
        only the position of the central wavelength will be altered.
        '''
        vm=VoigtModel()
        par_v=vm.guess(self.bkgd, x=self.lamb)
        par_v['center'].set(value=532e-9, vary=True)
        if sigma is not None: #if a width is provided, fix it.
            par_v['sigma'].set(value=sigma, vary=False)
        if gamma is not None: #if a width is provided, fix it.
            par_v['gamma'].set(value=gamma, vary=False, expr='')
        elif gamma is None: #vary gamma for better fit - this is not done by default
            par_v['gamma'].set(value=par_v['sigma'].value,vary=True, expr='')

        ##Fit the Voigt Model to the data
        if weights is True:
            weights=self.bkgd/self.bkgd_err
        if weights is False:
            weights=np.ones_like(self.bkgd)
        self.vm_fit=vm.fit(self.bkgd,par_v,x=self.lamb, weights=weights)
        self.l0=self.vm_fit.best_values['center']
        self.sigma=self.vm_fit.best_values['sigma']
    def symmetric_crop_around_l0(self):
        #now crop the data so that the response is symmetric for the convolution to work
        l0_i=find_nearest(self.lamb, self.l0)
        take_l=min(l0_i,self.lamb.size-l0_i)-1 #trim the shortest distance from the central wavelength
        low_i, high_i=l0_i-take_l, l0_i+take_l+1 # odd length array with peak at centre.
        self.lamb=self.lamb[low_i:high_i]
        self.bkgd=self.bkgd[low_i:high_i]
        self.shot=self.shot[low_i:high_i]
        self.shot_err=self.shot_err[low_i:high_i]
        self.bkgd_err=self.bkgd_err[low_i:high_i]
        #the response is taken from the model so it is nice and smooth
        self.response=self.vm_fit.best_fit[low_i:high_i]
        self.shift=self.lamb-self.l0 #this is useful for plotting data
    def fit_fibre(self, pp, interpolation_scale=1, notch=None, weights=True):
        '''
        Fit the shot data. This is complicated!
        This examines the dictionary provided, determines which are dependent and independent variables
        It then chooses the correct model, and sets up lmfit.
        '''
        self.pp_valid={}
        self.iv_dict={} #dictionary for independent variables

        if pp['model'] is 'nLTE':
            valid_keys=['model','n_e','T_e','V_fe','A','T_i','V_fi','stray','amplitude', 'offset', 'shift']
        if pp['model'] is 'electron':
            valid_keys=['model','n_e','T_e','V_fe','stray','amplitude', 'offset', 'shift']
        for k in valid_keys:
            self.pp_valid[k]=pp[k]
        # if n_e is set to None, use the value read in from a file
        if self.pp_valid['n_e'][0] is None:
            self.pp_valid['n_e']=(self.n_e, self.pp_valid['n_e'][1])
        #sort into dictionaries based on another dictionary
        for k,v in self.pp_valid.items():
            if v[1] is True: #independent variable
                self.iv_dict[k]=self.pp_valid[k][0]#get first element of list

        interpolated_lambda=add_points_evenly(self.lamb, interpolation_scale)
        interpolated_response=self.vm_fit.eval(x=interpolated_lambda)

        self.iv_dict['lambda_range']=interpolated_lambda
        self.iv_dict['interpolation_scale']=interpolation_scale
        self.iv_dict['lambda_in']=self.l0
        self.iv_dict['response']=interpolated_response
        self.iv_dict['theta']=self.theta
        if pp['model'] is 'nLTE':
            self.iv_dict['Z_Te_table']=generate_ZTe_table(pp['A'][0])
            skw_func=Skw_nLTE_stray_light_convolve
        if pp['model'] is 'electron':
            skw_func=Skw_e_stray_light_convolve

        if notch is None:
            self.iv_dict['notch']=np.ones_like(self.shot)
        if notch is not None:
            self.iv_dict['notch']=notch

        skw=Model(skw_func, independent_vars=list(self.iv_dict.keys())) #create our model with our set variables.
        #our best guesses at what the fitting parameters should be
        for k,v in self.pp_valid.items():
            if v[1] is False: #dependent variable
                try:
                    skw.set_param_hint(k, value = v[0], min=v[2]) #if a minimum is provided, use it
                except IndexError:
                    skw.set_param_hint(k, value = v[0])
        '''multiply shot by notch for fitting purposes'''
        if notch is None:
            shot=self.shot
        else:
            shot=self.shot*notch
        '''now do the fitting'''
        if weights is True:
            weights=self.shot/self.shot_err
        if weights is False:
            weights=np.ones_like(self.shot)
        self.skw_res=skw.fit(shot,verbose=False, **self.iv_dict, weights=weights)
        # get a dictionary of parameters used for the fit
        self.gather_parameters()
    def gather_parameters(self):
        params=self.skw_res.best_values.copy()
        for k,v in self.iv_dict.items():
            params[k]=v
        [params.pop(k, None) for k in ['lambda_range','lambda_in','interpolation_scale','response', 'Z_Te_Table']]#remove pointless keys
        try:
            params.pop('notch', None)
        except:
            pass

#        try:
#            Te=self.skw_res.best_values['T_e']
#        except KeyError: #if this is an independent variable it isn't in the fit
#            Te=self.pp_valid['T_e'][0]
#        params['Te']=Te
#        try:
#            Ti=self.skw_res.best_values['T_i']
#        except KeyError: #if this is an independent variable it isn't in the fit
#            Ti=self.pp_valid['T_i'][0]
#        params['Ti']=Ti
#        params['n_e']=self.pp_valid['n_e'][0]
        self.params=params
        self.params['model']=self.pp_valid['model']
        if self.params['model'] is 'nLTE':
            #self.params['Z']=Z_nLTE(self.params['T_e'], self.iv_dict['Z_Te_table'])
            self.params['Z']=float(self.iv_dict['Z_Te_table'](self.params['T_e']))


    def calculate_alpha(self):
        lambda_De=7.43*(self.params['T_e']/self.params['n_e'])**0.5 #in m
        th=self.theta*np.pi/180
        k=4*np.pi*np.sin(th/2.0)/self.l0
        self.params['alpha']=np.abs(1/(k*lambda_De))
    def calculate_intensity_form_factor(self):
        #calculate the expected TS intensity from the form factor equation
        self.params['Z']=Z_nLTE(self.params['T_e'], self.iv_dict['Z_Te_table'])
        Z=self.params['Z']
        alpha=self.params['alpha']
        TS_norm=Z*self.params['n_e']*alpha**4/((1+alpha**2)*(1+alpha**2+alpha**2*Z*self.params['T_e']/self.params['T_i']))
        self.params['predicted intensity FF']=TS_norm
        return TS_norm
    def calculate_intensity_without_stray(self, return_spectrum = False):
        #calculate the expected TS intensity from the fitted data, without stray light
        pars = self.params.copy()
        pars['interpolation_scale'] = 1 # actually we use the interpolation scale from the fit, this is a dummy
        pars['lambda_in'] = self.iv_dict['lambda_in'] 
        pars['lambda_range'] = self.iv_dict['lambda_range']
        pars['response'] = self.iv_dict['response'] # use interpolated response for convolution
        pars['offset'] = 0
        pars['stray'] = 0
        pars['notch'] = np.ones_like(pars['response'])
        pars.pop('model')
        pars.pop('Z')
        pars.pop('alpha')

        skw = Skw_nLTE_stray_light_convolve(**pars)

        if return_spectrum is True:
            shift_interp = (pars['lambda_range'] - pars['lambda_in'])*1e10
            return shift_interp, skw
        else:
            dlambda=pars['lambda_range'][1]-pars['lambda_range'][0]
            self.params['predicted intensity no stray'] = skw.sum()*dlambda
            return skw.sum()*dlambda
        
    def export_data(self, filename):
        data=list(zip(self.shift*1e10, self.bkgd, self.response, self.shot, self.shot_err, self.skw_res.best_fit))
        headings=('Wavelength shift', 'Background', 'Response', 'Shot', 'Shot error', 'Fit')
        units=('Angstroms', 'a.u.', 'a.u.', 'a.u.', 'a.u.', 'a.u.')
        with open(filename+'.dat', 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headings)
            writer.writerow(units)
            for f in data:
                writer.writerow(f)


class TS_Analysis:
    def __init__(self, folder, shot, backgrounds, calibration=False, skip_footer=0):
        '''
        Loads .asc files.
        shot should be the path to a single.asc
        backgrounds is a list of paths to multiple backgrounds - useful if you want to sum several together
        calibration is if your .asc files don't have the wavelength calibration embdeed,
        you can borrow it from another. .asc.
        skip_footer is if the end of your file is corrupted I guess? But maybe don't use it.
        '''
        ts_dir=os.getcwd()
        os.chdir(folder)
        self.s_name=os.path.basename(shot)[:8]
        data=np.genfromtxt(open(shot,"rb"),delimiter="\t", skip_footer=skip_footer)
        self.shot=np.rot90(data[:,1:-1]) #shot data
        self.x_axis=data[:,0]#x_axis, either pixel number or wavelength scale
        #Create an empty array to load multiple backgrounds into
        self.background=np.zeros((self.shot.shape))
        #load multiple backgrounds and sum them together
        for b in backgrounds:
            data=np.genfromtxt(open(b,"rb"),delimiter="\t", skip_footer=skip_footer)
            bkgd=np.rot90(data[:,1:-1])
            self.background+=bkgd
        #If a file is provided with a seperate calibration, use that calibration
        if calibration:
            data=np.genfromtxt(open(calibration,"rb"),delimiter="\t")
            self.x_axis=data[:,0]
        try:
            self.n_e=np.genfromtxt(self.s_name+' n_e at fibres.txt', delimiter=',', usecols=[1])
        except OSError:
            print('No electron density file found, enter electron densities manually.')
        os.chdir(ts_dir)

    def plot_fibre_edges(self, spacing=18.0, offset=8):
        '''
        Plots the intensity of the signal on each fibre.
        And also little red dots at where the fiber edges are currently meant to be.
        '''
        pe=self.shot[:,950:1100].sum(1)
        self.fig_edge, self.ax_edge=plt.subplots(figsize=(15,4))
        self.plot_edge=self.ax_edge.plot(pe)
        fe=np.arange(offset, pe.size-1, spacing)
        fe=np.append(fe,pe.size-1)
        fe=np.around(fe)
        fe=fe.astype(int)#need integers to index arrays
        #can't have the first bin start before the image
        fe[fe<=0]=0
        self.fibre_edges=fe
        self.plot_dots=self.ax_edge.plot(fe, pe[fe], 'r.')
    def find_fibre_edges(self):
        interact(self.plot_fibre_edges, spacing=(10,60,0.1),offset=(0,512,1))
    def split_into_fibres(self, discard_rows=6, dark_counts = None, dark_x=300, intensity_x=60, fibres = 28):
        '''Splits the images into 1D arrays for each fibre'''
        fe=np.array(self.fibre_edges)
        fe=fe[:fibres+1] # remove non-fibres
        self.N_fibres=fe.size
        ##zero dark counts
        if dark_counts is None:
            dark_counts=(self.shot[:, :dark_x].mean()+self.shot[:, -dark_x:].mean())/2
            self.dark_counts_shot=dark_counts
            s=self.shot-dark_counts
            dark_counts=(self.background[:, :dark_x].mean()+self.background[:, -dark_x:].mean())/2
            self.dark_counts_bkgd=dark_counts
            b=self.background-dark_counts
        else:
            s=self.shot-dark_counts
            b=self.background-dark_counts

        ##Shot Fibres
        shot_fibres=np.zeros((fe.size-1, self.shot.shape[1]))
        shot_abs_err=np.zeros_like(shot_fibres)
        #take the data in each fibre, discard some rows where there is cross talk
        #and then sum together in the y direction to improve signal to noise.
        for i in np.arange(fe.size-1):
            rs=s[fe[i]+discard_rows:fe[i+1]-discard_rows,:]
            m_x=rs.shape[1]//2
            intensity=rs[:,m_x-intensity_x:m_x+intensity_x].sum(1)
            n_intensity=intensity/intensity.mean()
            d_normed=np.array([rs[i]/n_intensity[i] for i in range(intensity.size)])
            d_wavg, d_wstd=weighted_avg_and_std(d_normed, axis=0, weights=n_intensity)

            shot_fibres[i] = d_wavg
            shot_abs_err[i] = d_wstd
        self.shot_fibres=shot_fibres
        self.shot_abs_err=shot_abs_err 
        ##Background Fibres
        bkgd_fibres=np.zeros((fe.size-1, self.background.shape[1]))
        bkgd_abs_err=np.zeros_like(bkgd_fibres)

        for i in np.arange(fe.size-1):
            rs=b[fe[i]+discard_rows:fe[i+1]-discard_rows,:]
            m_x=rs.shape[1]//2
            intensity=rs[:,m_x-intensity_x:m_x+intensity_x].sum(1)
            n_intensity=intensity/intensity.mean()
            d_normed=np.array([rs[i]/n_intensity[i] for i in range(intensity.size)])
            d_wavg, d_wstd=weighted_avg_and_std(d_normed, axis=0, weights=n_intensity)

            bkgd_fibres[i]= d_wavg
            bkgd_abs_err[i] = d_wstd
        self.bkgd_fibres=bkgd_fibres
        self.bkgd_abs_err=bkgd_abs_err
    def zero_fibres(self, lower=500, upper=1500, dark_counts=None):
        '''self.shot_fibres_z=np.zeros((self.N_fibres,upper-lower))
        self.bkgd_fibres_z=np.zeros((self.N_fibres,upper-lower))
        self.S_T=np.zeros(self.N_fibres)
        for fin, f in enumerate(self.shot_fibres):
            #remove offset due to dark counts
            if dark_counts is None:
                mean1=f[0:300].mean()
                mean2=f[-300:].mean()
                mean=(mean1+mean2)/2
            else:
                mean=dark_counts
            f=f-mean #zero the fibres
            self.S_T[fin]=np.trapz(f, x=self.x_axis) #calculate the total scattered amplitude
            f=f[lower:upper]
            self.shot_fibres_z[fin]=f
        for fin, f in enumerate(self.bkgd_fibres):
            #remove offset due to dark counts
            if dark_counts is None:
                mean1=f[0:300].mean()
                mean2=f[-300:].mean()
                mean=(mean1+mean2)/2
            else:
                mean=dark_counts
            f=f-mean #zero the fibres
            f=f[lower:upper]'''
        self.shot_fibres_z=self.shot_fibres[:,lower:upper]
        self.bkgd_fibres_z=self.bkgd_fibres[:,lower:upper]
        self.shot_abs_err=self.shot_abs_err[:,lower:upper]
        self.bkgd_abs_err=self.bkgd_abs_err[:,lower:upper]
        self.x_axis=self.x_axis[lower:upper]
    def pair_fibres(self, angle_a, angle_b):
        fibre_angles=angle_a+angle_b
        l=self.x_axis
        params=list(zip(self.bkgd_fibres_z,self.shot_fibres_z, self.bkgd_abs_err, self.shot_abs_err, fibre_angles))
        self.fibres=[Fibre(l,bkgd,shot,bkgd_err,shot_err, angle) for bkgd,shot,bkgd_err,shot_err, angle in params]
        self.fibres_a=self.fibres[:len(angle_a)]
        self.fibres_b=self.fibres[len(angle_a):]
        try:
            for fa,fb,nee in zip(self.fibres_a, self.fibres_b, self.n_e):
                fa.n_e=nee
                fb.n_e=nee
        except AttributeError:
            pass
    def copy_background(self, good, bad):
        self.fibres[bad].bkgd=self.fibres[good].bkgd.copy() #copy a good background to overwrite a bad one
    def select_fibre(self, Fnum, Fset):
        if Fset=='A':
            f=self.fibres_a[Fnum-1]
        elif Fset=='B':
            f=self.fibres_b[Fnum-1]
        return f
    def plot_data(self, Fnum, Fset, sr=8, tm=1.0):
        '''Probably the prettiest plot you've ever seen'''
        f=self.select_fibre(Fnum,Fset)
        text_mul=tm
        fig, ax=plt.subplots(figsize=(16,10))
        bk_norm=0.5*f.shot.max()/f.bkgd.max()

        ax.step(f.lamb, bk_norm*f.bkgd, c='black', where='mid', label='Background')
        ax.step(f.lamb, f.shot, c='orange', where='mid', label='Shot', lw=3)
        ax.fill_between(f.lamb, y1=f.shot-f.shot_err, y2=f.shot+f.shot_err, step='mid', color='orange', alpha=0.5)

        #plotting region
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r'Wavelength (nm)',fontsize=20*text_mul)
        ax.set_ylabel('Intensity (a.u.)',fontsize=20*text_mul)
        ax.tick_params(labelsize=20*text_mul, pad=5, length=10, width=2)
        title_str=self.s_name+': Thomson Scattering for fibre '+str(Fnum)+Fset+r', $\theta=$'+str(f.theta)+r'$^{\circ}$'
        ax.set_title(title_str,fontsize=20*text_mul)
        ax.legend(fontsize=18*text_mul)
        plt.tight_layout()
        self.fig=fig
        self.ax=ax
    def pretty_plot(self, Fnum, Fset ,sr=8.0, tm=1.0, style='steps'):
        """Prints the output of the fit in a very pretty way. Plot options are available.

        Arguments:
            Fnum {float} -- Number of the fibre in a set
            Fset {float} -- Fibre set, A or B

        Keyword Arguments:
            sr {float} -- Shift Range, the number of Angstroms shift to plot around the centre (default: {8})
            tm {float} -- Text multiplier, to change the text size(default: {1.0})
            style {str} -- Style, either 'dots' or 'steps' (default: {'dots'})
        """
        f=self.select_fibre(Fnum,Fset)
        try:
            shift=f.params['shift']
        except KeyError:
            shift=0
        response=np.interp(f.lamb+shift, f.lamb, f.response)
        text_mul=tm

        bk_norm=0.5*f.shot.max()/f.bkgd.max()
        fig, ax=plt.subplots(figsize=(12,6))
        if style is 'dots':
            ax.plot(f.shift*1e10,bk_norm*f.bkgd, label='Background', lw=1, marker='o', color='gray')
            ax.plot(f.shift*1e10,bk_norm*response, label='Response', lw=1, ls='--', color='black')
            ax.scatter(f.shift*1e10,f.shot,label='Data', marker='o',lw=1, color='blue', alpha=0.5)
            ax.plot(f.shift*1e10,f.skw_res.best_fit, label='Best Fit', lw=2, ls='--', color='red')
        if style is 'steps':
            ax.step(f.shift*1e10,bk_norm*f.bkgd, label='Background', lw=1,  color='gray', where='mid')
            ax.step(f.shift*1e10,bk_norm*response, label='Response', lw=1, color='black', where='mid')
            ax.step(f.shift*1e10, f.shot, c='orange', where='mid', label='Data', lw=3)
            ax.fill_between(f.shift*1e10, y1=f.shot-f.shot_err, y2=f.shot+f.shot_err, step='mid', color='orange', alpha=0.5)
            ax.step(f.shift*1e10,f.skw_res.best_fit, label='Best Fit', lw=3, color='red', where='mid')
        #plotting region
        ax.set_ylim(bottom=0.0)
        ax.set_xlim([-sr,sr])
        ax.set_xticks(np.arange(-sr,sr+1,2))
        ax.set_xlabel(r'Wavelength shift, $(\AA)$',fontsize=10*text_mul)
        ax.set_ylabel('Intensity (a.u.)',fontsize=10*text_mul)
        ax.tick_params(labelsize=10*text_mul, pad=5, length=10, width=2)
        kms=r' $km\,s^{-1}$'
        if f.params['model'] is 'electron':
            string_list=[
                    r'$n_e= $'+str_to_n(f.params['n_e']/1e17,2)+r'$\times$10$^{17} cm^{-3}$',
                    r'$T_e= $'+str_to_n(f.params['T_e'],2)+' $eV$',
                    r'$V_{fe}= $'+str_to_n(f.params['V_fe']/1e3,2)+kms,
                    r'$\alpha\,= $'+str_to_n(f.params['alpha'],2),
                    ]

        if f.params['model'] is 'nLTE':
            string_list=[
                    r'$A\,= $'+str(f.params['A']),
                    r'$Z\,= $'+str_to_n(f.params['Z'],2),
                    r'$n_e= $'+str_to_n(f.params['n_e']/1e17,2)+r'$\times$10$^{17} cm^{-3}$',
                    r'$T_e= $'+str_to_n(f.params['T_e'],2)+' $eV$',
                    r'$T_i= $'+str_to_n(f.params['T_i'],2)+' $eV$',
                    r'$V_{fi}= $'+str_to_n(f.params['V_fi']/1e3,2)+kms,
                    r'$V_{fe}= $'+str_to_n(f.params['V_fe']/1e3,2)+kms,
                    r'$\alpha\,= $'+str_to_n(f.params['alpha'],2),
                    ]

        text_str=''
        for st in string_list:
            text_str=text_str+st+'\n'
        text_str=text_str[:-1]

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='gray', alpha=0.2)

        # place a text box in upper left in axes coords
        ax.text(0.02, 0.96, text_str, transform=ax.transAxes, fontsize=10*text_mul,
            verticalalignment='top', bbox=props)

        title_str=self.s_name+': Fit of Thomson Scattering for fibre '+str(Fnum)+Fset+r', $\theta=$'+str(f.theta)+r'$^{\circ}$'
        ax.set_title(title_str,fontsize=10*text_mul)
        ax.legend(fontsize=10*text_mul)
        plt.tight_layout()
        self.fig=fig
        self.ax=ax

    def pretty_plot14(self, Fset,sr=8.0, tm=0.8, style='steps'):
        """Prints the output of the fit in a legible way for comparing many fibres. Plot options are available.

        Arguments:
            Fset {float} -- Fibre set, A, B, Both1 (both sets side by side 1 to 7) or Both2 (both sets side by side 8 to 14)

        Keyword Arguments:
            sr {float} -- Shift Range, the number of Angstroms shift to plot around the centre (default: {8})
            tm {float} -- Text multiplier, to change the text size(default: {1.0})
            style {str} -- Style, either 'dots' or 'steps' (default: {'dots'})
        """
        
        fig, axs = plt.subplots(7, 2,figsize=(13,20))
        for i in range(0,14):
            if Fset == 'A' or Fset == 'B':
                if i < 7:
                    j=i
                    k=0
                    n=i+1
                    F_set=Fset
                else:
                    j=i-7
                    k=1
                    n=i+1
                    F_set=Fset
            elif Fset == 'Both1':
                if i < 7:
                    j=i
                    k=0
                    n=i+1
                    F_set='A'
                else:
                    j=i-7
                    k=1
                    n=i-6
                    F_set='B'
            elif Fset == 'Both2':
                if i < 7:
                    j=i
                    k=0
                    n=i+8
                    F_set='A'
                else:
                    j=i-7
                    k=1
                    n=i+1
                    F_set='B'

            ax=axs[j,k]
            f=self.select_fibre(n,F_set)

            try:
                shift=f.params['shift']
            except KeyError:
                shift=0
            response=np.interp(f.lamb+shift, f.lamb, f.response)
            text_mul=tm

            bk_norm=0.5*f.shot.max()/f.bkgd.max()
            #fig, ax=plt.subplots(figsize=(12,6))
            if style is 'dots':
                ax.plot(f.shift*1e10,bk_norm*f.bkgd, label='Background', lw=1, marker='o', color='gray')
                ax.plot(f.shift*1e10,bk_norm*response, label='Response', lw=1, ls='--', color='black')
                ax.scatter(f.shift*1e10,f.shot,label='Data', marker='o',lw=1, color='blue', alpha=0.5)
                ax.plot(f.shift*1e10,f.skw_res.best_fit, label='Best Fit', lw=2, ls='--', color='red')
            if style is 'steps':
                ax.step(f.shift*1e10,bk_norm*f.bkgd, label='Background', lw=1,  color='gray', where='mid')
                ax.step(f.shift*1e10,bk_norm*response, label='Response', lw=1, color='black', where='mid')
                ax.step(f.shift*1e10, f.shot, c='orange', where='mid', label='Data', lw=3)
                ax.fill_between(f.shift*1e10, y1=f.shot-f.shot_err, y2=f.shot+f.shot_err, step='mid', color='orange', alpha=0.5)
                ax.step(f.shift*1e10,f.skw_res.best_fit, label='Best Fit', lw=3, color='red', where='mid')
            #plotting region
            ax.set_ylim(bottom=0.0)
            ax.set_xlim([-sr,sr])
            ax.set_xticks(np.arange(-sr,sr+1,2))
            if j == 6:
                ax.set_xlabel(r'Wavelength shift, $(\AA)$',fontsize=10*text_mul)
            ax.set_ylabel('Intensity (a.u.)',fontsize=10*text_mul)
            ax.tick_params(labelsize=10*text_mul, pad=5, length=10, width=2)
            kms=r' $km\,s^{-1}$'
            if f.params['model'] is 'electron':
                string_list=[
                        r'$n_e= $'+str_to_n(f.params['n_e']/1e17,2)+r'$\times$10$^{17} cm^{-3}$',
                        r'$T_e= $'+str_to_n(f.params['T_e'],2)+' $eV$',
                        r'$V_{fe}= $'+str_to_n(f.params['V_fe']/1e3,2)+kms,
                        r'$\alpha\,= $'+str_to_n(f.params['alpha'],2),
                        ]

            if f.params['model'] is 'nLTE':
                string_list=[
                        r'$A\,= $'+str(f.params['A']),
                        r'$Z\,= $'+str_to_n(f.params['Z'],2),
                        r'$n_e= $'+str_to_n(f.params['n_e']/1e17,2)+r'$\times$10$^{17} cm^{-3}$',
                        r'$T_e= $'+str_to_n(f.params['T_e'],2)+' $eV$',
                        r'$T_i= $'+str_to_n(f.params['T_i'],2)+' $eV$',
                        r'$V_{fi}= $'+str_to_n(f.params['V_fi']/1e3,2)+kms,
                        r'$V_{fe}= $'+str_to_n(f.params['V_fe']/1e3,2)+kms,
                        r'$\alpha\,= $'+str_to_n(f.params['alpha'],2),
                        ]

            text_str=''
            for st in string_list:
                text_str=text_str+st+'\n'
            text_str=text_str[:-1]

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='gray', alpha=0.2)

            # place a text box in upper left in axes coords
            ax.text(0.02, 0.96, text_str, transform=ax.transAxes, fontsize=10*text_mul,
                verticalalignment='top', bbox=props)

            title_str=self.s_name+': Fit of Thomson Scattering for fibre '+str(n)+F_set+r', $\theta=$'+str(f.theta)+r'$^{\circ}$'
            ax.set_title(title_str,fontsize=10*text_mul)
            ax.legend(fontsize=10*text_mul)
            plt.tight_layout()
            self.fig=fig
            self.ax=ax
    def export_data(self, Fnum, Fset):
        f=self.select_fibre(Fnum,Fset)
        filename=self.s_name+' fit dat files/'+self.s_name+'_'+str(Fnum)+Fset+'_data_and_fit'
        f.export_data(filename)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def generate_ZTe_table(A):
    if A is 12:
        T_e, Z =  np.genfromtxt('zb_C.dat', delimiter='       ', skip_header=4, usecols = [0,1], unpack = True)
        Z_mod = CubicSpline(T_e, Z)
        return Z_mod
    if A is 27:
        T_e, Z =  np.genfromtxt('zb_Al.dat', delimiter=' ', skip_header=2, usecols = [0,1], unpack = True)
        Z_mod = CubicSpline(T_e, Z)
        return Z_mod
    if A is 64:
        T_e, Z = np.genfromtxt('zb_Cu.dat', delimiter='     ', skip_header=2, usecols = [0,2], unpack = True)
        Z_mod = CubicSpline(T_e, Z)
        return Z_mod
    if A is 183:
        T_e, Z = np.genfromtxt('zb_W.dat', delimiter=' ', usecols = [0,1],unpack = True)
        Z_mod = CubicSpline(T_e, Z)
        return Z_mod
    else:
        print("No data available for A:", A)

def ZTe_finder(n_e, ZTe_experimental, Z_guess, element='Al'):
    if element=='Al':
        Z_Te_table=np.genfromtxt('zb_Al.dat', delimiter=' ', skip_header=2)
        ni=np.array([1e17,5e17,1e18,5e18,1e19])
    if element=='C':
        Z_Te_table=np.genfromtxt('zb_C.dat')
        ni=np.array([1e19])#always choose lowest for now
    ni_guess=n_e/float(Z_guess)
    ind=find_nearest(ni, ni_guess)+1
    ZTe_list=Z_Te_table[:,0]*Z_Te_table[:,ind]
    index=np.where(ZTe_list>=ZTe_experimental)[0][0]
    Te=Z_Te_table[index,0]
    Z=Z_Te_table[index,ind]
    return Z, Te

def Z_finder(n_e, Te_experimental, Z_guess=4, element='Al'):
    if element=='Al':
        Z_Te_table=np.genfromtxt('zb_Al.dat', delimiter=' ', skip_header=2)
        ni=np.array([1e17,5e17,1e18,5e18,1e19])
    if element=='C':
        Z_Te_table=np.genfromtxt('zb_C.dat')
        ni=np.array([1e19])#always choose lowest for now
    ni_guess=n_e/float(Z_guess)
    ind=find_nearest(ni, ni_guess)+1
    Te_list=Z_Te_table[:,0]
    index=np.where(Te_list>=Te_experimental)[0][0]
    Z=Z_Te_table[index,ind]
    return Z

def add_points_evenly(initial_array, scale):
    return np.linspace(initial_array[0], initial_array[-1], initial_array.size*scale-scale+1)

def weighted_avg_and_std(values, weights, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=axis, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, axis=axis, weights=weights)
    return (average, np.sqrt(variance))
