# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:42:59 2017

@author: jdhare
"""
from datetime import date, timedelta
from scopes import ScopeChannel, MitlBdots, MachineDiagnostics
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")
start_date=date(2018,8,3)
end_date=date(2018,8,18)

delta = end_date - start_date         # timedelta
all_shots=[]
for i in range(delta.days + 1):
    date=start_date + timedelta(days=i)
    shot=date.strftime('s%m%d_%y')
    try:
        ScopeChannel(shot, "3", 'A1')
    except (FileNotFoundError, UserWarning):
        pass
    else:
        all_shots.append(shot)
bad_shots=['s1215_17'] 
shots = [x for x in all_shots if x not in bad_shots]      
#%%
md=[MachineDiagnostics(s) for s in shots]
s=[m.calculate_LGS_times() for m in md]
spreads=[m.LGS_spread for m in md]
peak_current=[m.calculate_peak_current(mitl_bdot=3) for m in md]
#%%
fig,ax=plt.subplots()
ax.scatter(spreads,peak_current)
ax.set_xlim([0,200])
#%%
scope="3"
tm=[ScopeChannel(s, scope, 'C2') for s in shots]
TM_times=[np.round(tmm.time[np.where(tmm.data>0.5)[0][0]]) for tmm in tm]

marx='Z'
scope="11"
LGT_volts={'G':'A1', 'H':'A2','C':'B1', 'Z':'B2'}

lg=[ScopeChannel(s, scope, LGT_volts[marx]) for s in shots]
LGT_times=[np.round(lgg.time[np.where(lgg.data>2)[0][0]]) for lgg in lg]

mb=[]

for s in shots:
    m=MitlBdots(s)
    m.truncate()
    m.integrate()
    mb.append(m)
    
Peak_I=[int(np.abs(m.mbds[3].B).max()) for m in mb]

scope="10"
LGS_dIdt={'G':'A1', 'H':'A2','C':'B1', 'Z':'B2'}

LGS=[[ScopeChannel(s, scope, LGS_dIdt[switch]) for switch in LGS_dIdt] for s in shots]
LG_times=[[np.round(l.time[np.where(l.data>0.5)[0][0]]) for l in lg] for lg in LGS]
LG_spread=[np.max(l)-np.min(l) for l in LG_times]
    
#%%
from matplotlib import cm

start = 0.2
stop = 1.0
number_of_lines= len(shots)
cm_subsection = np.linspace(start, stop, number_of_lines) 
colors = [ cm.YlOrBr(x) for x in cm_subsection ]

fig, ax=plt.subplots(2,1,figsize=(8,8))

for tmm,c in zip(tm, colors):
    ax[0].plot(tmm.time, tmm.data, label=tmm.shot, c=c)
    
ax[0].set_xlim(100,500)
#ax.set_ylim(0,5)
ax[0].legend()

for lgg,c in zip(lg, colors):
    ax[1].plot(lgg.time, lgg.data, label=lgg.shot, c=c)
    
ax[1].set_xlim(1200,1600)
ax[1].set_ylim(0,5)
ax[1].legend()

#%%
for s in shots:
    m=MitlBdots(s)
    m.truncate()
    m.integrate()
    mb.append(m)
    
Peak_I=np.array([[int(np.abs(mm.B).max()) for mm in m.mbds] for m in mb])

fig,ax=plt.subplots()
ax.scatter(Peak_I[:,1],Peak_I[:,3])
ax.set_xlim([200,350])
