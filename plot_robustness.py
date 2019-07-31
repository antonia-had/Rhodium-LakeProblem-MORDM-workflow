import matplotlib.pyplot as plt
import numpy as np
from j3 import J3


robustness_dps = np.loadtxt("robustness_dps.txt")*100
robustness_IT = np.loadtxt("robustness.txt")*100

#fig = plt.figure(figsize=(12,7))
#ax = fig.add_subplot(1,1,1)
#ax.fill_between(range(len(robustness_dps)),np.sort(robustness_dps)[::-1],color='#08519c')
#ax.fill_between(range(len(robustness_IT)),np.sort(robustness_IT)[::-1],color='#a50f15')
#ax.set_ylim([0,100])
#ax.set_ylabel('Percent of Sampled SOWs in which Criteria are Met',fontsize=16)
#ax.set_xlabel('Solution # (sorted by rank)',fontsize=16)
#ax.tick_params(axis='both',labelsize=14)
#plt.savefig('robustness_comparison.png')
#plt.close()

dps_output=load("dps_output.csv")[1]
output=load("output.csv")[1]

for i in range(len(output)):
    output[i]['strategy']=1
    output[i]['robustness']=robustness_IT[i]
for i in range(len(dps_output)):
    dps_output[i]['strategy']=0
    dps_output[i]['robustness']=robustness_dps[i]

merged = DataSet(output+dps_output)

J3(merged.as_dataframe(list(['max_P', 'utility', 'inertia', 'reliability', 'strategy', 'robustness'])))