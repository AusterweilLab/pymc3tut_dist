import pymc3 as pm 
import numpy as np
import matplotlib.pyplot as plt

# Seed the rng for exact reproducible results
seed = 68492
np.random.seed(seed)

## Generate data with K components.
# Means, standard deviations, proportions
mus = [ 0,  6,-5]
sigmas = [ 1, 1.5, 3]
ps  = [.2, .5,.3]
# Total amount of data
N = 1000
# Stack data into a single array
y = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)),
               np.random.normal(mus[1], sigmas[1], int(ps[1]*N)),
               np.random.normal(mus[2], sigmas[2], int(ps[2]*N))])

## Build list of models 
# We want a range of models ranging from K=1 to K=3
K_range = range(3) 
# Initialize lists
gmm_list = [] 
trace_list = []
# Specify some sampling options
draws = 2000
nchains = 4

# Loop through K_range, building model and sampling each time
for ki in K_range:
    # Add a new Model object to the list
    gmm_list += [pm.Model()] 
    K = ki+1 # Here ki is the index, K is the number of groups the model assumes

    # Name each model for easy identification during comparison
    gmm_list[ki].name = '%d-Group' % K
    
    with gmm_list[ki]: 
        # Prior over z - only applicable if K>1
        if K>1:
            p = pm.Dirichlet('p', a=np.array([1.]*K))

            # z is the component that the data point is being sampled from.
            # Since we have N data points, z should be a vector with N elements.    
            z = pm.Categorical('z', p=p, shape=N)    

        # Prior over the component means and standard deviations
        mu = pm.Normal('mu', mu=0., sd=10., shape=K)     
        sigma = pm.HalfCauchy('sigma', beta=1., shape=K) 

        # Specify the likelihood        
        if K>1:
            Y_obs = pm.Normal('Y_obs', mu=mu[z], sd=sigma[z], observed=y)            
            # Specify the sampling algorithms to use
            step1 = pm.NUTS(vars=[p, mu, sigma])
            step2 = pm.ElemwiseCategorical(vars=[z])
            steps = [step1,step2]
        else:
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)    
            # Specify the sampling algorithms to use - don't include z or p  because they don't exist when K==1
            step1 = pm.NUTS(vars=[mu, sigma])
            steps = [step1]
            
        # Start the sampler, and save results in a new element of trace_list
        trace_list += [pm.sample(draws=draws,       
                                 nchains=nchains,
                                 step=steps,
                                 random_seed=seed)]
    
## Model comparison
# Convert model and trace into dictionary pairs
dict_pairs = dict(zip(gmm_list,trace_list))
# Perform WAIC (or LOO by setting ic='LOO') comparisons
compare = pm.compare(dict_pairs, ic='WAIC')
# Print comparisons
print(compare)
