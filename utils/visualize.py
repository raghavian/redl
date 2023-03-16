import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import matplotlib

params = {'font.size': 30,
          'font.sans-serif': 'cmr10',
          'font.weight': 'bold',
          'axes.labelsize':34,
          'axes.titlesize':40,
#          'axes.labelweight':'heavy',
#          'axes.titleweight':'bold',
          'legend.fontsize': 32,
          'xtick.major.pad':'10'
         }
matplotlib.rcParams.update(params)
df = pd.read_csv('downsample.csv')
plt.figure(figsize=(10,10))
scales = df.scale.unique()
labels = ['8b\_Opt','AMP','Half']
indices = [[0,0,0],[1,1,0],[1,0,1]]
labels = ['Baseline','8bit Opt+AMP','8bit Opt+half']
seeds = [1,10,100]

for idx in range(len(indices)): 

    i = indices[idx]
#    acc = df[df.bnb==i[0]][df.amp==i[1]][df.half==i[2]].acc.values
#    pdb.set_trace()
    acc_mean = np.sort(np.array([df[df.bnb==i[0]][df.amp==i[1]][df.half==i[2]][df.scale==s].acc.values.mean() for s in scales ]))[::-1]
    acc_std = np.sort(np.array([df[df.bnb==i[0]][df.amp==i[1]][df.half==i[2]][df.scale==s].acc.values.std() for s in scales ]))


#    df[df.bnb==i[0]][df.amp==i[1]][df.half==i[2]].groupby(['seed']).mean().acc.values
#    acc_std = df[df.bnb==i[0]][df.amp==i[1]][df.half==i[2]].groupby(['seed']).std().acc.values
    plt.plot(scales,acc_mean,label=labels[idx],linewidth=4)
    plt.fill_between(scales,acc_mean-acc_std,acc_mean+acc_std,alpha=0.3)
plt.legend(loc='lower left')
plt.xlabel('Downsampling factor')
plt.ylabel('Test Perf.')
plt.xticks(scales)
plt.tight_layout()
plt.savefig('downsample.pdf',dpi=300)
if 0:
    df = pd.read_csv('data_50yr.csv')
    P = len(df.patient_id.unique())
    print('Found %d subjects'%P)
    age = df.groupby(['patient_id']).mean().age.values
    outcomes = df.BIRADS.values
    plt.figure(figsize=(10,10))
    #plt.subplot(121)
    plt.hist(age,bins=20)
    plt.xlabel('Subject Age (yr)')

if 0:
    plt.subplot(122)
    plt.hist((outcomes == 0)*1.0)
    plt.xticks([0,1])
    plt.xlabel('Follow-up (no:0,yes:1)')


