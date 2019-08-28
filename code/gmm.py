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

## Plot the data as a histogram
plt.hist(y, bins=20, alpha=0.5)
# Add axes labels
plt.xlabel('Simulated values')
plt.ylabel('Frequencies')
plt.show() 

## Build model 
gmm = pm.Model() 
# Specify number of groups
K = 3

with gmm: 
    # Prior over z 
    p = pm.Dirichlet('p', a=np.array([1.]*K))

    # z is the component that the data point is being sampled from.
    # Since we have N data points, z should be a vector with N elements.    
    z = pm.Categorical('z', p=p, shape=N)    

    # Prior over the component means and standard deviations
    mu = pm.Normal('mu', mu=0., sd=10., shape=K)     
    sigma = pm.HalfCauchy('sigma', beta=1., shape=K) 

    # Specify the likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu[z], sd=sigma[z], observed=y)    
    
## Run sampler
with gmm:    
    # Specify the sampling algorithms to use
    step1 = pm.NUTS(vars=[p, mu, sigma])
    step2 = pm.ElemwiseCategorical(vars=[z])

    # Start the sampler!
    trace = pm.sample(draws=2000,
                      nchains=4,
                      step=[step1, step2],
                      random_seed=seed) 

# Plot results
pm.traceplot(trace, 
             varnames=['mu','p','sigma'], # Specify which variables to plot
             lines={'mu':mus,'p':ps,'sigma':sigmas}) # Plots straight lines - useful for simulations
plt.show()

## Posterior Predictive Checks  
# Obtain posterior samples
pp = pm.sample_ppc(model=gmm, trace=trace)
# Plot original data
plt.hist(y, bins=20, alpha=0.5)
# Plot posterior predictives on top of that 
plt.hist(np.random.choice(pp['Y_obs'].flatten(),size=len(y)), bins=20, alpha=0.5)
# Add legend and axes labels
plt.legend(['Data','Predictions'])
plt.xlabel('Simulated values')
plt.ylabel('Frequencies')
plt.show() 

