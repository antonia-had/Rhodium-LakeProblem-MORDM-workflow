'''
This script is a copy of this one https://github.com/Project-Platypus/Rhodium/blob/master/examples/Basic/example.py
by Dave Hadka, with minor changes for training purposes. 
'''

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *

# Construct the lake problem
def lake_problem(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(pollution_limit)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (max_P, utility, inertia, reliability)

model = Model(lake_problem)

# Define all parameters to the model that we will be studying
model.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

# Define the model outputs
model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

# Define any constraints (can reference any parameter or response by name)
model.constraints = [Constraint("reliability >= 0.95")]

# Some parameters are levers that we control via our policy
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# Some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]

# Prepare the cache for storing intermediate results
setup_cache(file="example.cache")

# Optimize the model or get cached results if they exist.  Note that the
# call to optimize is wrapped in a lambda function to enable lazy evaluation.
output = cache("output", lambda: optimize(model, "NSGAII", 10000))

#Save optimization results 
output.save("output.csv") 
#Save only the objectives from the optimization results 
output.as_dataframe()[list(model.responses.keys())].to_csv('output_objectives.csv')

for i in range(len(output)):
    output[i]['strategy']=1
    dps_output[i]['strategy']=0

merged = DataSet(output+dps_output)

J3(merged.as_dataframe(list(dps_model.responses.keys())+['strategy']))

SOWs = sample_lhs(model, 1000)
reevaluation = [evaluate(model, update(SOWs, policy)) for policy in output]
reevaluation_dps = [evaluate(dps_model, update(SOWs, policy)) for policy in dps_output]

for i in range(len(reevaluation)):
    reevaluation[i].save("reevaluation_"+str(i)+".csv")
    reevaluation_dps[i].save("reevaluation_dps_"+str(i)+".csv")

robustness = np.zeros(len(output))
robustness_dps = np.zeros(len(dps_output))

for i in range(len(robustness)):
    robustness[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation[i]])
    robustness_dps[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation_dps[i]])
    
for i in range(len(robustness)):
    robustness[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation[i]])
    robustness_dps[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation_dps[i]])
 
colnames = ['sol_no']+list(dps_model.responses.keys())+['strategy']    
merged_sorted = load('overallreference.csv', names=colnames)[1]