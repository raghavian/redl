import pandas as pd
import numpy as np
import glob
import pdb

LOC='LOG_DIR/'

files = sorted(glob.glob(LOC+'*.txt'))
N = len(files)
print('Found %d files'%N)

cols = ['method','bnb','amp','half','scale','seed','acc','loss','param','gpu','time','energy','c02','dist']

C = len(cols)
df = pd.DataFrame(columns=cols,index=np.arange(N)).fillna(0)
to_sec = np.array([3600,60,1])
for fIdx in range(N):
    f = files[fIdx].split('/')[-1].split('.')[0]
    if 'bnb' in f:
        df.loc[fIdx,'bnb'] = 1
    if 'amp' in f:
        df.loc[fIdx,'amp'] = 1
    if 'half' in f:
        df.loc[fIdx,'half'] = 1
    df.loc[fIdx,'seed'] = int(f.split('_S_')[-1][:3])
    df.loc[fIdx,'method'] = f.split('_S_')[0].split('_')[3]
    df.loc[fIdx,'scale'] = int(f.split('_D_')[-1][:1])
    idx = -1
    print(fIdx,f)
    data = list(pd.read_csv(files[fIdx]).columns)
    data = [np.array(d.split(':')[1:],dtype=float) for d in data[:-1]]
    data[4] = [(data[4]*to_sec).sum()]
    for c in range(8):
        df.iloc[fIdx,6+c] = data[c][0]
df.loc[:,'energy'] = df.loc[:,'energy']*1e3

df.to_csv('results.csv',index=False)

print('Writing into tex file')
gdf = df.groupby(['method','bnb','amp','half']).mean().reset_index()
gdf[['method','bnb','amp','half','acc','param','gpu','time','energy']].to_latex('results_mean.tex',float_format="%.3f",index=False)
gdf = df.groupby(['method','bnb','amp','half']).std().reset_index()
gdf[['method','bnb','amp','half','acc','param','gpu','time','energy']].to_latex('results_std.tex',float_format="%.3f",index=False)
print('Done!')
