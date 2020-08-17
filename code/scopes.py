import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.integrate

class Globals:
    '''
    Defines the location of the scope folders. if you need to use a local version instead of LINNE, you can modify the location here.
    '''
    scope_folder=r"D:\\MAGPIE Scopes\\"

class ScopeChannel:
    '''
    Basic building block for dealing with scope data.
    
    Properties:
        shot: shot number as a string
        scope: scope number as a string
        channel: channel as a string
        time: time as a numpy array
        data: voltage as a numpy array
    
    '''
    def __init__(self, shot, scope, channel, folder = None):
        '''
        Args:
            shot: a string containing the shot number, in standard 'sMMDD_YY' format
            scope: number of the scope as a string, eg. "1"
            channel: scope channel as a string eg. "C1"
        
        '''
        self.shot=shot
        self.scope=scope
        self.channel=channel

        if folder is None:
            folder = Globals.scope_folder
        fn=folder +"scope"+scope+"_"+shot
        self.fn=fn
        self.time=np.loadtxt(fn+"time")
        self.data=np.loadtxt(fn + "_" + channel)[1:]
        
        
class MitlBdots:
    '''
    Groups four MITL b-dot scope channels together, providing routines for intregrating signals to get a measure of the peak current.
    Properties:
        mbds: list of ScopeChannels for each of the 4 MITL b-dots
        
    Each ScopeChannel in mbds gains additional properties through truncate and integrate:
    Prorperties after truncate:
        time_tr: array of times, truncated
        time_tr0: array of times, truncated and zeroed to the current start
        data_tr: data truncated
    Properties after integrate:
        B: integrated voltage. No constant of proportionality for area here, this is a relative measure of B field and hence current.
        time_B: time series for plotting B
        time_B0: zerod time series for plotting B.
    '''
    def __init__(self, shot):
        '''
        Args:
            shot: shot number as a string, in standard 'sMMDD_YY' format
        '''
        self.shot=shot
        scope="3"
        mitl_bdots=['A1','A2','B1','B2']
        self.mbds=[ScopeChannel(shot, scope, m) for m in mitl_bdots]
    def truncate(self, threshold=1.0, window=1000):
        '''
        Determines the start time by looking for when the signal passes above a certain voltage,
        then truncates the signal before that and after a certain window has passed.
        Determines any systematic offset to the voltage using data from the start of the record,
        and subtracts this from the truncated data.
        Args:
            threshold: voltage which the signal much exceed to signify the start of the current pulse
            window: number of data points after the start time to include in integration
        '''
        for m in self.mbds:
            start=np.nonzero(abs(m.data)>threshold)[0][0]
            start=start-100
            if start<0:
                start=0
            m.time_tr=m.time[start:start+window]
            m.time_tr0=m.time_tr-m.time_tr[50] #50 data points found sufficient at threshold =1.0
            zero=np.mean(m.data[0:start])
            m.data_tr=m.data[start:start+window]-zero
    def integrate(self):
        '''
        Cumulative trapezoidal integration of the truncated data,
        creating a new time series and truncated time series that is
        one element shorter than the previous arrays as integration results in a
        shorter array.
        '''
        for m in self.mbds:
            m.B=scipy.integrate.cumtrapz(m.data_tr,m.time_tr)
            m.time_B=m.time_tr[:-1]
            m.time_B0=m.time_tr0[:-1]

                
class BdotPair:
    '''
    Handles pairs of bdots to produce magnetic field. Intended for use by the
    Bdots class below, as opposed to directly.
    Properties:
        bd1, bd2: two ScopeChannel objects containing the scope data
    '''
    def __init__(self, shot, scope="1", bdot1='A1', bdot2='A2'):
        '''
        Args:
            shot: shot number as a string, in standard 'sMMDD_YY' format
            scope: string containing number of scope
            bdot1, bdot2: strings containing the channels of the scopes.
        '''
        self.shot=shot
        #bdot signals 1 and 2
        self.bd1=ScopeChannel(shot, scope, bdot1)
        self.bd2=ScopeChannel(shot, scope, bdot2)
    def zero(self):
        '''
        Removes systematic offset in voltage using first element of array
        '''
        self.bd1.data=self.bd1.data-self.bd1.data[0]
        self.bd2.data=self.bd2.data-self.bd2.data[0]
    def truncate(self, threshold=1.0, window=1000, cal=[1,1], fix_start=None):
        '''
        Args:
            Threshold: voltage to be exceeded to signify start of signal
            Window: number of points to be taken
            Cal: list of two calibrations to convert voltage to magnetic field. Includes effective area, attenuators etc.
            fix_start: manually specify a start time, ignoring the threshold.
        '''
        #find the start of the current pulse with a  high threshold
        sig1=self.bd1.data
        start=np.nonzero(abs(sig1)>threshold)[0][0]
        #back off a bit so we can see the zero signal
        self.start=start-100
        if fix_start is not None:
            self.start=find_nearest(self.bd1.time,fix_start)
        self.time=self.bd1.time[self.start:self.start+window]
        self.bd1_tr=self.bd1.data[self.start:self.start+window]*cal[0]
        self.bd2_tr=self.bd2.data[self.start:self.start+window]*cal[1]
        self.add()
        self.subtract()
    def add(self):
        '''
        Calculate the electrostatic (sum) component of the voltage
        '''
        self.estat=(self.bd1_tr+self.bd2_tr)/2.0      
    def subtract(self):
        '''
        Calculate the inductive (difference) component of the voltage.
        '''
        self.dBdt=(self.bd1_tr-self.bd2_tr)/2.0
    def integrate(self):
        '''
        Cumulative trapezoidal integration of the truncated data,
        creating a new time series and truncated time series that is
        one element shorter than the previous arrays as integration results in a
        shorter array.
        '''
        self.B=scipy.integrate.cumtrapz(self.dBdt,self.time)/1e9
        self.time_B=self.time[:-1]
    def plot(self, data, ax=None, flip=1, bdname=None):
        '''
        Plotting functions for the varies properties
        Args:
            data: a string to determine what to plot.
                    raw: raw data
                    tr: truncated data
                    sum_diff: inductive and electrostatic components
                    B: integrated magnetic field
            flip: whether to flip the second channel
            bdname: a string with the bdot names in the standard"T89" format for "T8" and "T9" as the names.
        '''
        if ax is None:
            fig, ax=plt.subplots()
        if bdname is not None:
            b1=bdname[0:2]
            b2=bdname[0]+bdname[2]
        if data is "raw":
            t=self.bd1.time
            d1=self.bd1.data
            d2=self.bd2.data
            l1=b1+' raw'
            l2=b2+' raw'
        if data is "tr":
            t=self.time
            d1=self.bd1_tr
            d2=self.bd2_tr
            l1=b1+' truncated'
            l2=b2+' truncated'
        if data is "sum_diff":
            t=self.time
            d1=self.estat
            d2=self.dBdt
            l1=bdname+' Electrostatic'
            l2=bdname+' dB/dt'
        if data is "B":
            t=self.time_B
            d1=self.B
            d2=None
            l1=bdname+' Magnetic Field'
        ax.plot(t, flip*d1, label=l1, lw=4)
        if d2 is not None:
            ax.plot(t, flip*d2, label=l2, lw=4)
        ax.legend()
        
class Bdots:
    '''
    A group of pairs of bdots from the same shot.
    Properties:
        bd: list of BdotPairs
    '''
    def __init__(self, shot, pairs, attenuations, diameters, scope="1", threshold=1.0, window=1000, fix_start=None):
        '''
        Args:
            shot: shot number as a string, in standard 'sMMDD_YY' format
            pairs: a dictionary of names:channel eg. {"T12":"A", "T34":B}
            attenuations: a dictionary of channels:attenuations eg. {"A1":10.3, "A2": 8.6}
            diameters: a dictionary of diameters in mm, same for both bdots eg. {"T12": 1.0, "T34":0.5}
            scope: string containing number of scope
            threshold: the voltage which must be reached to singify the start of the signal
            window: number of data points after the start to integrate over
            fix_start: fix at a given time, rather than using the threshold technique
        '''
        self.shot=shot
        self.bd={}
        for k, v in  pairs.items():
            bd1=v+"1"
            bd2=v+"2"
            area=(1e-3*diameters[k]/2.0)**2*np.pi
            calibration=[attenuations[bd1]/area, attenuations[bd2]/area]
            self.bd[k]=BdotPair(shot, scope, bdot1=bd1, bdot2=bd2)
            self.bd[k].zero()
            self.bd[k].truncate(threshold=threshold,cal=calibration, window=window, fix_start=fix_start)
            self.bd[k].integrate()
    def plot(self, name, data, ax=None, flip=1):
        self.bd[name].plot(data, ax, flip, bdname=name)
    def plot_raw(self, name):
        self.bd[name].plot_raw()
    def plot_estat_dBdt(self, name):
        self.bd[name].plot_estat_dBdt()
    def plot_B(self, name):
        self.bd[name].plot_B()
            
            
class Rogowskis:
    '''
    Class to deal with Rogowski coil measurements of current.
    Modified version of BdotPairs
    Properties:
        bd1, bd2: ScopeChannel corresponding to the two rogowskis.
    '''
    def __init__(self, shot):
        '''
        Args:
            shot: shot number as a string, in standard 'sMMDD_YY' format
        '''
        self.shot=shot
        #rogowski 1 and 2      
        self.bd1=ScopeChannel(shot, '2', 'c1')
        self.bd2=ScopeChannel(shot, '2', 'c2')
    def truncate(self, threshold=0.2, window=1000, cal=[10*10.4*3e9,-10.48*10.79*3e9]):
        '''
        Truncates record to start of signal to avoid integrating over irrelevant section of record
        Args:
            threshold: voltage which signifies the start of the signal
            window: number of data points to integrate over after start of signal
            cal: calibration of Rogowskis, including attenuators and geometric calibration
        '''
        #find the start of the current pulse with a  high threshold
        sig1=self.bd1.data
        start=np.nonzero(abs(sig1)>threshold)[0][0]
        #back off a bit so we can see the zero signal
        self.start=start-50
        self.time=self.bd1.time[self.start:self.start+window]
        z1=np.mean(self.bd1.data[0:200]) #zero the data
        z2=np.mean(self.bd2.data[0:200])
        self.bd1_tr=(self.bd1.data[self.start:self.start+window]-z1)*cal[0]
        self.bd2_tr=(self.bd2.data[self.start:self.start+window]-z2)*cal[1]
    def integrate(self, return_posts=8, min_signal=5e4):
        '''
        Integrates Rogowski signals to determine current pulse
        Args:
            return_posts: number of return posts
            min_signal: ignore a rogowski if it doesn't produce more than this signal, which usually indicates a fault in that rogowski'
        '''
        self.I1=scipy.integrate.cumtrapz(self.bd1_tr,self.time)/1e9
        self.I2=scipy.integrate.cumtrapz(self.bd2_tr,self.time)/1e9
        #check currents are positive:
        if np.abs(self.I1.max())<np.abs(self.I1.min()):
            self.I1=-self.I1
        if np.abs(self.I2.max())<np.abs(self.I2.min()):
            self.I2=-self.I2
        #check that tehre's signal
        if self.I2.max()<min_signal:
            self.I_Tot=self.I1*return_posts
            print(self.shot+": using Rog 1 only")
        if self.I1.max()<min_signal:
            self.I_Tot=self.I2*return_posts
            print(self.shot+": using Rog 2 only")
        if self.I1.max()>5e4 and self.I2.max()>5e4:
            self.I_Tot=(self.I1+self.I2)*return_posts/2.0
            print(self.shot+": using both Rogs")
        self.time_I=self.time[:-1]
        self.t0=self.time_I[np.where(self.I_Tot>2e3)[0][0]]
        self.time_0ed=self.time_I-self.t0
        self.Imax=self.I_Tot.max()/1e6
    def plot(self, data, ax=None, scale=1):
        '''
        Sundry plotting functions.
        Args:
            data: a string to determine what to plot.
                raw: raw data
                tr: truncated data
                I: separate currents for the two rogowskis
                I_Tot: summed currents
                I_Tot0: summed currents, plotted with timescale zerod
        scale: arbitrary scaling factor (can't recall why this is useful...)
        '''
        if ax is None:
            fig, ax=plt.subplots()
        if data is "raw":
            t=self.bd1.time
            d1=self.bd1.data
            d2=self.bd2.data
            l1='R1 raw'
            l2='R2 raw'
        if data is "tr":
            t=self.time
            d1=self.bd1_tr
            d2=self.bd2_tr
            l1='R1 truncated'
            l2='R2 truncated'
        if data is "I":
            t=self.time_I
            d1=self.I1
            d2=self.I2
            l1='R1 Current'
            l2='R2 Current'
        if data is "I_Tot":
            t=self.time_I
            d1=self.I_Tot
            d2=None
            l1=self.shot+' Current'
        if data is "I_Tot0":
            t=self.time_0ed
            d1=self.I_Tot
            d2=None
            l1=self.shot+' Current'
        ax.plot(t, scale*d1, label=l1, lw=4)
        if d2 is not None:
            ax.plot(t, scale*d2, label=l2, lw=4)
        ax.legend()
        
class MachineDiagnostics:
    '''
    Grabs data from several scope channels to calculate machine performance
    Properties:
        TM: ScopeChannel for trigger marx
        LGT: ScopeCHannel for line gap trigger
        m: MitlBdots for this shot
        LGS: list of ScopeChannels for the line gap switch dI/dt measurements.
    '''
    marx='Z'
    LGS_channels={'G':'A1', 'H':'A2','C':'B1', 'Z':'B2'}
    def __init__(self, shot, LGT_channel='Z'):
        '''
        Args:
            shot: shot number as a string, in standard 'sMMDD_YY' format
            LGT_channel: optional argument used to choose which LGT channel to use, useful if one of the cables broke for this shot
        '''
        self.TM=ScopeChannel(shot, "3", 'C2')
        self.LGT=ScopeChannel(shot, "11", self.LGS_channels[LGT_channel])
        self.m=MitlBdots(shot)
        self.LGS=[ScopeChannel(shot, "10", self.LGS_channels[switch]) for switch in ['G','H','C','Z']]
        self.shot=shot
    def calculate_peak_current(self, mitl_bdot=3):
        '''
            Integrate one of the Mitl b-dots to get a relative measure of the peak current.
            Args:
                mitl_bdot: numbner from 0 to 3 to select Mitl bdot to integrate.
        '''
        try:
            self.m.truncate()
        except IndexError:
            return None
        self.m.integrate()
        self.Peak_I=int(np.abs(self.m.mbds[mitl_bdot].B).max())
        return self.Peak_I
    def calculate_LGS_times(self, threshold=0.5):
        '''
        Calculates the start of the LGS rise times for each switch, and determines the spread.
        Args:
            threshold: voltage which signifies signal start
        '''
        self.LGS_times=[]
        for lg in self.LGS:
            '''
            Sometimes there is no data on the scope, so if this fails we ignore that LGS
            '''
            try:
                t=find_time_for_threshold(lg, threshold) 
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                pass
            else:
                self.LGS_times.append(t)
        self.LGS_spread=np.max(self.LGS_times)-np.min(self.LGS_times)
        return self.LGS_times,self.LGS_spread
    def calculate_LGT_time(self, threshold=2):
        '''
        Determine when the line gap trigger pulse reached the VTL
        Args:
            threshold: voltage which signifies signal start
   
        '''
        self.LGT_time=find_time_for_threshold(self.LGT, threshold)
        return self.LGT_time
    def calculate_TM_time(self, threshold=0.5):
        '''
        Determine when the trigger marx pulse began
        Args:
            threshold: voltage which signifies signal start
   
        '''
        self.TM_time=find_time_for_threshold(self.TM, threshold)
        return self.TM_time
        
def find_nearest(array,value):
    '''
    helper function to find index of maximum of an array.
    Args:
        array: array to search
        value: value to find
    Returns:
        index of array element closest to value
    '''
    idx = (np.abs(array-value)).argmin()
    return idx

def find_time_for_threshold(scope_channel, threshold):
    '''
    Helper function to find first time at which a scope exceeds some threshold.
    Args:
        scope_channel: a ScopeChannel object to search
        threshold: voltage to be exceeded
    Returns:
        time rounded to 1 ns of the first time when the scope data exceeds the threshold
    '''
    return np.round(scope_channel.time[np.where(scope_channel.data>threshold)[0][0]])
