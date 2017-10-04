import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import colormaps as cmaps
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
import os
import image_registration as ir
from skimage.measure import profile_line
import imreg_dft as ird
import images2gif as ig
import pickle
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display

#for backwards compatibility reasons, odl notebooks might refer to eg. FaradayMap2, so let's link them here:
FaradayMap2=FaradayMap
PolarimetryMap2=PolarimetryMap
NeLMap2=NeLMap

class DataMap:
    def __init__(self, flip_lr, rot_angle, multiply_by, scale):
        if flip_lr is True:
            self.d=np.fliplr(self.d)
        if rot_angle is not None:
            self.d=rotate(self.d, rot_angle)
        self.rot_angle=rot_angle
        self.data=self.d*multiply_by
        self.scale=scale
        self.s_name=os.path.basename(self.fn)[:8]
    def plot_data_px(self, clim=None, multiply_by=1, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        d=self.data*multiply_by
        return ax.imshow(d, clim=clim, cmap=self.cmap)
    def set_origin(self, origin, extent):
        self.origin=origin
        ymin=origin[0]-extent[1]*self.scale
        ymax=origin[0]-extent[0]*self.scale
        xmin=origin[1]+extent[2]*self.scale
        xmax=origin[1]+extent[3]*self.scale
        self.origin_crop=(extent[1]*self.scale,-extent[2]*self.scale)
        self.data_c=self.data[ymin:ymax, xmin:xmax]
        self.extent=extent[2:4]+extent[0:2]
    def plot_data_mm(self, clim=None, multiply_by=1, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        d=self.data_c*multiply_by
        return ax.imshow(d, cmap=self.cmap, interpolation='none', clim=clim, extent=self.extent, aspect=1)
    def plot_contours_px(self, levels=None, multiply_by=1, ax=None, color='k'):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        d=self.data*multiply_by
        return ax.contour(d, levels, origin='image', hold='on',colors=color)
    def plot_contourf_mm(self, levels=None, multiply_by=1, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        d=self.data_c*multiply_by
        return ax.contourf(d, levels=levels, cmap=self.cmap,extent=self.extent,origin='image',aspect=1)
    def create_lineout(self, start=(0,0), end=(0,0), lineout_width=20, verbose=False):
        '''
        start and end are in mm on the grid defined by the origin you just set
        '''
        #find coordinates in pixels
        start_px=self.mm_to_px(start)
        end_px=self.mm_to_px(end)
        if verbose is True:
            print(start_px,end_px)
        #use scikit image to do a nice lineout on the cropped array
        self.lo=profile_line(self.data_c, start_px,end_px,linewidth=lineout_width)
        #set up a mm scale centred on 0
        px_range=self.lo.size/2
        self.mm=np.linspace(-px_range, px_range, 2*px_range)/self.scale #flip range to match images
    def plot_lineout(self, ax=None, label='', multiply_by=1):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        ax.plot(self.mm, self.lo*multiply_by, label=label, lw=4)        
    def mm_to_px(self,mm):
        scale=self.scale
        px_origin=self.origin_crop
        return (int(-mm[0]*scale+px_origin[0]),int(mm[1]*scale+px_origin[1]))
    def pickle_transform(self, fn):
        try:
            pickle.dump(self.transform, open(fn, 'wb'))
        except:
            print('No Transform found!')
    
class NeLMap(DataMap):
    def __init__(self, filename, scale, multiply_by=1, flip_lr=False, rot_angle=None):
        self.fn=filename[:8]
        d=np.loadtxt(open(filename,"r"),delimiter=",")
        d=d-np.nan_to_num(d).min()
        d=np.nan_to_num(d)
        if flip_lr is True:
            d=np.fliplr(d)
        if rot_angle is not None:
            d=rotate(d, rot_angle)
        self.data=d*multiply_by
        self.scale=scale
        self.cmap=cmaps.cmaps['inferno']
        
class Interferogram(DataMap):
    def __init__(self, filename, scale, flip_lr=False, rot_angle=None):
        self.fn=filename[:8]
        d=plt.imread(filename)
        if flip_lr is True:
            d=np.fliplr(d)
        if rot_angle is not None:
            d=rotate(d, rot_angle)
        self.data=d
        self.scale=scale
    def plot_data_px(self, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        return ax.imshow(self.data)
    def plot_data_mm(self, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        d=self.data_c
        return ax.imshow(d, interpolation='none', extent=self.extent, aspect=1)

class PolarimetryMap(DataMap):
    def __init__(self, R0fn, R1fn, B0fn, B1fn, S0fn, S1fn, rot_angle=None):
        self.fn=R0fn[:8]
        self.rot_angle=rot_angle
        self.R0=plt.imread(R0fn)
        self.R1=np.fliplr(plt.imread(R1fn))
        self.B0=plt.imread(B0fn)
        self.B1=np.fliplr(plt.imread(B1fn))
        self.S0=plt.imread(S0fn)
        self.S1=np.fliplr(plt.imread(S1fn))
        if rot_angle is not None:
            self.R0=rotate(self.R0, rot_angle)
            self.R1=rotate(self.R1, rot_angle)
            self.B0=rotate(self.B0, rot_angle)
            self.B1=rotate(self.B1, rot_angle)
            self.S0=rotate(self.S0, rot_angle)
            self.S1=rotate(self.S1, rot_angle)
        #normalise registration images
        R0s=self.R0.sum()
        R1s=self.R1.sum()
        self.R0=self.R0*R1s/R0s
        self.cmap='seismic'
    def register(self, constraints=None, transform=None):
        if transform is None:
            t=ird.similarity(self.R0, self.R1, numiter=3, constraints=constraints)
            transform = { your_key: t[your_key] for your_key in ['angle','scale','tvec'] }
        self.transform=transform
        self.RT=ird.transform_img_dict(self.R1, self.transform)
        self.BT=ird.transform_img_dict(self.B1, self.transform)
        self.ST=ird.transform_img_dict(self.S1, self.transform)
    def nudge_transform(self):        
        def plot_transform(scale, angle, tx, ty, limits):
            imT=ird.imreg.transform_img(self.R1, scale=scale, angle=angle, tvec=(ty,tx))
            diff=self.R0-imT
            fig, ax=plt.subplots(figsize=(10,8))
            ax.imshow(diff, cmap='bwr', clim=[-limits,limits])

        ty=widgets.FloatSlider(min=self.transform['tvec'][0]-50,
                               max=self.transform['tvec'][0]+50,
                               step=1,
                               description='Translate in y:',
                               value=self.transform['tvec'][0],
                                continuous_update=False)

        tx=widgets.FloatSlider(min=self.transform['tvec'][1]-50,
                       max=self.transform['tvec'][1]+50,
                       step=1,
                       description='Translate in x:',
                       value=self.transform['tvec'][1],
                               continuous_update=False)
        scale=widgets.FloatSlider(min=0,
                               max=2,
                               step=0.05,
                               description='Scale:',
                               value=self.transform['scale'],
                                  continuous_update=False)
        angle=widgets.FloatSlider(min=-180,
                       max=180,
                       step=0.5,
                       description='Angle (radians):',
                       value=self.transform['angle'],
                                  continuous_update=False)

        limits=widgets.FloatSlider(min=0,
                                  max=1,
                                  step=0.01,
                                  value=0.1,
                                   description='Colourbar limits:',
                                   continuous_update=False)

        self.w=interactive(plot_transform,
                      scale=scale, 
                      angle=angle, 
                      tx=tx,
                      ty=ty,
                      limits=limits
                      )

        display(self.w)
    def confirm_nudge(self):
        wargs=self.w.kwargs
        self.transform['scale']=wargs['scale']
        self.transform['angle']=wargs['angle']
        self.transform['tvec']=(wargs['ty'],wargs['tx'])
        self.RT=ird.transform_img_dict(self.R1, self.transform)
        self.BT=ird.transform_img_dict(self.B1, self.transform)
        self.ST=ird.transform_img_dict(self.S1, self.transform)

    def convert_to_alpha(self, beta=3.0):
        self.N0=self.S0/self.B0
        self.N1=self.ST/self.BT
        diff=self.N0-self.N1
        self.diff=np.nan_to_num(diff)
        beta=beta*np.pi/180
        self.data=-(180/np.pi)*0.5*np.arcsin(self.diff*np.tan(beta)/2.0)
        
class InterferogramOntoAlpha(DataMap):
    def __init__(self, polmap, I0, I1):
        self.fn=I0[:8]
        I0=plt.imread(I0)
        self.I0s=np.sum(I0,2)
        self.pm=polmap
        #scale and flip registration to polarisation data
        R0=self.pm.R0
        scale=R0.shape[0]/self.I0s.shape[0]
        I0z=zoom(self.I0s, scale)
        crop=(I0z.shape[1]-R0.shape[1])/2
        I0zc=I0z[:,crop:crop-R0.shape[1]]
        self.I0zcn=np.flipud(I0zc/I0zc.max())
        #do the same to the inteferogram
        I1=plt.imread(I1)
        I1s=np.sum(I1,2)
        I1z=zoom(I1s, scale)
        I1zc=I1z[:,crop:crop-R0.shape[1]]
        self.I1zcf=np.flipud(I1zc)
        self.cmap='gray'
    def register(self, constraints=None, transform=None):
        if transform is None:
            t=ird.similarity(self.pm.R0, self.I0zcn, numiter=3, constraints=constraints)
            transform = { your_key: t[your_key] for your_key in ['angle','scale','tvec'] }
        self.transform=transform
        self.I0T=ird.transform_img_dict(self.I0zcn, self.transform)
        self.data=ird.transform_img_dict(self.I1zcf, self.transform)
    def plot_overlay_px(self, clim=None, ax=None, transparency=0.8):
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        ax.imshow(self.pm.data, cmap='RdBu', clim=clim)
        ax.imshow(self.data, cmap='gray', alpha=transparency)

class FaradayMap(DataMap):
    def __init__(self, polmap, I0, ne, flip_ne=False):
        self.fn=polmap.fn[:8]
        I0=plt.imread(I0)
        self.I0s=np.sum(I0,2)
        I1=np.loadtxt(ne, delimiter=',')
        I1=I1-np.nan_to_num(I1).min()
        self.I1=np.nan_to_num(I1)
        self.pm=polmap
        #flip cos interferometry camera images are upside down wrt faraday
        self.I0s=np.flipud(self.I0s)
        self.I1=np.flipud(self.I1)
        #rotate data to match polarisation map
        if self.pm.rot_angle is not None:
            self.I0s=rotate(self.I0s,self.pm.rot_angle)
            self.I1=rotate(self.I1,self.pm.rot_angle)
        #in order to perform image registration, the two images must be the same size
        #scale and flip to data
        R0=self.pm.R0
        scale_y=R0.shape[0]/self.I0s.shape[0]
        scale_x=R0.shape[1]/self.I0s.shape[1]
        
        if scale_y>scale_x:
            scale=scale_y
            I0z=zoom(self.I0s, scale)
            I1z=zoom(self.I1, scale)
            crop=(I0z.shape[1]-R0.shape[1])//2
            I0zc=I0z[:,crop:crop+R0.shape[1]]
            I1zc=I1z[:,crop:crop+R0.shape[1]]
        if scale_x>scale_y:
            scale=scale_x
            I0z=zoom(self.I0s, scale)
            I1z=zoom(self.I1, scale)
            crop=(I0z.shape[0]-R0.shape[0])//2
            I0zc=I0z[crop:crop+R0.shape[0],:]
            I1zc=I1z[crop:crop+R0.shape[0],:]
        self.I0z=I0z
        self.I0zcn=I0zc/I0zc.max()
        self.I1zc=I1zc
        if flip_ne is True:
            self.I1zc=np.flipud(self.I1zc)   
        self.cmap='seismic'
    def register(self, constraints=None, transform=None):
        if transform is None:
            t=ird.similarity(self.pm.R0, self.I0zcn, numiter=3, constraints=constraints)
            transform = { your_key: t[your_key] for your_key in ['angle','scale','tvec'] }
        self.transform=transform
        self.I0T=ird.transform_img_dict(self.I0zcn, self.transform)
        self.I1T=ird.transform_img_dict(self.I1zc, self.transform)
        self.data=5.99e18*self.pm.data/self.I1T
        
class OpticalFrames:
    def __init__(self, start, IF):
        self.load_images()
        self.normalise()
        self.start=start
        self.IF=IF
        self.frame_times=np.arange(start, start+12*IF, IF)
    def load_images(self):
        shot=os.path.split(os.getcwd())[-1][0:8] #automatically grab the shot number
        b=[]
        s=[]
        for i in range(1,13):
            if i<10:
                st="0"+str(i)
            else:
                st=str(i) 
            bk_fn=shot+" Background_0"+st+".png"
            bk_im=plt.imread(bk_fn) #read background image
            #bk_im=np.asarray(np.sum(bk_im,2), dtype=float)
            b.append(bk_im)#np.asarray(np.sum(bk_im,2), dtype=float)) #convert to grrayscale
            sh_fn=shot+" Shot_0"+st+".png" 
            sh_im=plt.imread(sh_fn)
            s.append(sh_im)
           
        self.shot=shot
        self.b=b
        self.s=s
    def normalise(self):
        norms=[b_im[100:-100,100:-100].sum() for b_im in self.b]
        n_max=max(norms)
        nn=[n/n_max for n in norms]
        self.s_n=[s_im[100:-100,100:-100]/n for s_im, n in zip(self.s, nn)]
    def logarithm(self, lv_min=-4, lv_max=0.2):
        self.s_l=[np.log(s_im) for s_im in self.s_n]
        self.s_nl=[(np.clip(s_im, a_min=lv_min, a_max=lv_max)-lv_min)/(lv_max-lv_min) for s_im in self.s_l]
    def rotate(self, angle_deg=0):
        self.s_r=[rotate(s_im, angle_deg)for s_im in self.s_nl]
    def crop(self, origin, xcrop=400, ycrop=400):
        x0=origin[1]
        y0=origin[0]
        self.origin=[ycrop,xcrop]
        self.s_c=[s_im[y0-ycrop:y0+ycrop,x0-xcrop:x0+xcrop] for s_im in self.s_r]
    def plot(self, array, frame=1, clim=None, ax=None):
        fin=frame-1
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        ax.imshow(array[fin], cmap='afmhot', clim=clim)
        ax.axis('off')
        ax.set_title('t='+str(self.frame_times[fin])+' ns', fontsize=22)
    def plot_norm(self, frame=1, clim=None, ax=None):
        self.plot(self.s_n, frame=frame, clim=clim, ax=ax)
    def plot_log(self, frame=1, clim=None, ax=None):
        self.plot(self.s_nl, frame=frame, clim=clim, ax=ax)
    def plot_rot(self, frame=1, clim=None, ax=None):
        self.plot(self.s_r, frame=frame, clim=clim, ax=ax)
    def plot_crop(self, frame=1, clim=None, ax=None):
        self.plot(self.s_c, frame=frame, clim=clim, ax=ax)
    def plot_sequence(self, array=None, frames=list(range(1,13)), clim=None, figsize=None):
        xframes=round(len(frames)/2)
        if array is None:
            array=self.s_c
        if figsize is None:
            figsize=(xframes*4,16)
        fig, ax=plt.subplots(2,xframes, figsize=figsize)
        ax=ax.flatten()
        for fin, f in enumerate(frames):
            fn=f-1 #shift to 0 indexed arrays
            a=ax[fin]
            a.imshow(array[fn], cmap='afmhot', clim=clim)
            a.axis('off')
            a.set_title('t='+str(self.frame_times[fn])+' ns', fontsize=22)
        fig.suptitle("Optical Framing images from "+self.shot, fontsize=32)
        fig.tight_layout(w_pad=0, h_pad=0)
        self.fig=fig
    def save_sequence(self, filename=None):
        if filename is None:
            filename=self.shot+" frame sequence"
        self.fig.savefig(filename+".png")        
    def create_lineout(self, axis=0, frame=1,centre=None,average_over_px=20, mm_range=10, scale=29.1, ax=None):
        px_range=mm_range*scale
        fn=frame-1 #shift to 0 indexed arrays
        if axis is 1:
            d=np.transpose(self.s_c[fn])
            y0=self.origin[1] if centre is None else centre
            x0=self.origin[0]
        if axis is 0:
            d=self.s_c[fn]
            y0=self.origin[0] if centre is None else centre
            x0=self.origin[1]
        section=d[y0-average_over_px:y0+average_over_px, x0-px_range:x0+px_range]
        self.lo=np.mean(section, axis=0)
        self.mm=np.linspace(-px_range, px_range, self.lo.size)/scale
        if ax is None:
            fig, ax=plt.subplots(figsize=(12,8))
        ax.plot(self.mm, self.lo, label='t='+str(self.frame_times[fn])+' ns', lw=4)
    def save_gif(self, filename, clim, width=6):
        w=width
        h=w/self.s_c[0].shape[1]*self.s_c[0].shape[0]
        fig, ax=plt.subplots(figsize=(w,h))
        hot_im=[]
        for im in self.s_c:
            ax.imshow(im, cmap='afmhot', clim=clim)
            plt.axis('off')
            fig.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0,
                wspace=0, hspace=0)
            fig.canvas.draw()
            w,h=fig.canvas.get_width_height()
            buf=np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape=(h,w,3)
            hot_im.append(buf)
        ig.writeGif(filename+'.gif',hot_im, duration=0.2)
        
