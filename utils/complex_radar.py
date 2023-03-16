import matplotlib.pyplot as plt
import numpy as np
from utils.radar import ComplexRadar
import pdb
import pandas as pd

LOC = 'data/results.csv'

plt.rcParams["text.usetex"] = True

seed = 123
np.random.seed(seed)

# complex radar plot
format_cfg = {
    "rad_ln_args": {"visible": True},
    "outer_ring": {"visible": True},
    "angle_ln_args": {"visible": True},
    "rgrid_tick_lbls_args": {"fontsize": 24},
    "theta_tick_lbls": {"fontsize": 24},
    "theta_tick_lbls_pad": 24,
    'legend.fontsize': 20,
}
yellow = "#FFD700"
dark_yellow = "#FF8C00"
dark_green = "#006400"
dark_blue = "#00008B"
red = "#FF0000"
black = "#000000"
grey = "#808080"

df = pd.read_csv(LOC)
df = df.groupby(['method','bnb','amp','half']).mean().reset_index()
methods = df.method.unique()
cols = ['time','acc','energy','gpu']
#labels = ["$T$(s)", "$P_{v}$", "$E$(kWh)", "$GPU$(GB)"]

for m in methods:
    
    methDf = df[df.method == m].reset_index()
#    pdb.set_trace()
    ranges = [(methDf[c].min()*0.9,methDf[c].max()*1.) for c in cols]
    plt.clf()
    fig = plt.figure(figsize=(12,12))
    radar = ComplexRadar(fig, cols, ranges, n_ring_levels=5, show_scales=True, format_cfg=format_cfg)

    radar.plot(methDf.loc[0,cols].values.tolist(), color=dark_blue, linewidth=0.8)
    radar.fill(methDf.loc[0,cols].values.tolist(), alpha=0.4, color=dark_blue)

    radar.plot(methDf.loc[1,cols].values.tolist(), color=yellow, linewidth=0.8)
    radar.fill(methDf.loc[1,cols].values.tolist(), alpha=0.4, color=yellow)

    radar.plot(methDf.loc[2,cols].values.tolist(), color=dark_green, linewidth=0.8)
    radar.fill(methDf.loc[2,cols].values.tolist(), alpha=0.4, color=dark_green)
    if 'efficient' in m:
        radar.plot(methDf.loc[3,cols].values.tolist(), color=grey, linewidth=0.8)
        radar.fill(methDf.loc[3,cols].values.tolist(), alpha=0.4, color=grey)
    else:
        radar.plot(methDf.loc[4,cols].values.tolist(), color=grey, linewidth=0.8)
        radar.fill(methDf.loc[4,cols].values.tolist(), alpha=0.4, color=grey)

    radar.plot(methDf.loc[3,cols].values.tolist(), color=red, linewidth=0.8)
    radar.fill(methDf.loc[3,cols].values.tolist(), alpha=0.4, color=red)


    radar.use_legend(['_nolegend_','baseline','_nolegend_','AMP','_nolegend_','8bitOpt','_nolegend_','AMP+8bitOpt','_nolegend_','8bitOpt+Half'], fontsize=24,loc='upper right')

    plt.savefig(m+"_radar.pdf", dpi=300, bbox_inches="tight")

