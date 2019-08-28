import pymc3 as pm  
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import datetime
import os
from ellipse import plot_ellipse

# Seed the rng for exact reproducible results
seed = 68492
np.random.seed(seed)

## Load data
with open('data.p','rb') as f:
    data_all = pickle.load(f)

# Plot raw data
lim = 2.5 # Set the limits of the subplots
alpha = .5 # Set transparency of markers
markersize = 20 # Set size of markers
# Prepare subplots
fig, axx = plt.subplots(3,7,figsize=(25, 10))
# Create list of datasets containing only generated exemplars (not learned exemplars)
data = {}
for id,pid in enumerate(data_all.keys()):
    data[pid] = data_all[pid][20:]
# Loop through each participant and plot their data
for ip,pid in enumerate(data_all.keys()):
    # Identify an appropriate subplot
    ax = axx[np.unravel_index(ip, (3,7), 'C')]
    # Plot trained exemplars
    ax.scatter(data_all[pid][0:20, 0], data_all[pid][0:20, 1],s = markersize,color='blue',alpha=alpha,marker='o')
    # Plot novel exemplars
    ax.scatter(data_all[pid][20:, 0], data_all[pid][20:, 1],s = markersize,color='Red',alpha=alpha,marker='x')
    # Format axes
    ax.set_ylim(-lim,lim)
    ax.set_xlim(-lim,lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Add title
    ax.set_title(pid) 

# Specify number of participants and number of models
n_ppt = len(data)
n_models = 3
gmm_all = [pm.Model() for i in range(n_models)]

## Build models
for mi in range(n_models):
    print(f'Building Gaussian mixture model with {mi + 1} component(s).')
    gmm_all[mi].name = f'{mi + 1}-Group'  # Name each model for easy identification during comparison
    with gmm_all[mi]:
        k = mi+1
        if k>1:
            # Prior over component weights (only applicable with k>1)
            p = pm.Dirichlet('p', a=np.array([1.]*k), testval=np.ones(k)/k)
            
        # Prior over component means
        mus_p = [pm.MvNormal('mu_%d' % pid, 
                             mu=pm.floatX(np.zeros(2)),
                             tau=pm.floatX(0.1 * np.eye(2)),
                             shape=(k,2))
                 for pi,pid in enumerate(data.keys())]
        
        # Cholesky decomposed LKJ prior over component covariance matrices
        packed_L = [[pm.LKJCholeskyCov('packed_L_%d_%d' % (pid,i),
                                       n=2,
                                       eta=2.,
                                       sd_dist=pm.HalfCauchy.dist(.01))
                    for i in range(k)]
                    for pi,pid in enumerate(data.keys())]
        
        # Unpack packed_L into full array
        L = [[pm.expand_packed_triangular(2, packed_L[pi][i])
            for i in range(k)]
            for pi,pid in enumerate(data.keys())]
        
        # Convert L to sigma for convenience
        sigma = [[pm.Deterministic('sigma_%d_%d' % (pid,i) ,L[pi][i].dot(L[pi][i].T))
                for i in range(k)]
                for pi,pid in enumerate(data.keys())]

        # Specify the likelihood
        if k>1:
            mvnl = [[pm.MvNormal.dist(mu=mus_p[pi][i],chol=L[pi][i])
                    for i in range(k)]
                    for pi in range(n_ppt)]
            Y_obs = [pm.Mixture('Y_obs_%d' % pid,w=p, comp_dists=mvnl[pi],observed=data[pid])
                     for pi,pid in enumerate(data.keys())]
        else:
            Y_obs = [pm.MvNormal('Y_obs_%d' % pid,mu=mus_p[pi][0],chol=L[pi][0],
                                 observed=data[pid])
                     for pi,pid in enumerate(data.keys())] 
            
## Run sampler             
# Initialize list of traces
traces = []
draws = 2000
# Loop over each k-component model
for mi in range(n_models):
    print('Sampling from Model with {} component(s).'.format(mi+1))
    with gmm_all[mi]:
        step = pm.NUTS()
        
        #Start the sampler! For this larger dataset, it might take awhile -- allocate a few hours.
        traces += [pm.sample(draws, step=step,chains=4)]
        
        # Save the traces as a pickle so the hours spent running this are not lost. Warning: file can be very large
        with open('real_traces.p','wb') as f:
            pickle.dump(traces,f)
            
## Draw the fitted plots (with 95% density regions)
for ki in range(len(traces)):
    # Define the model index and number of components
    my_k_i = ki # Model index
    my_k = my_k_i+1 # Number of components 
    preamb_str = f'{my_k}-Group_' # Preamble of variable names
    # Extract relevant trace object
    trace = traces[my_k_i]
    # Specify the tail-end number of sample iterations to include
    num_samps_in = 500

    # Loop over number of models and make a separate plot each
    for mi in range(n_models):
        # Initialize subplots
        fig, ax2 = plt.subplots(3,7,figsize=(25, 10))
        for ip,pid in enumerate(data.keys()):
            # Get reference to current individual subplot to make it easier to work with
            ax = ax2[np.unravel_index(ip, (3,7), 'C')]
            
            # Get list of desired variable names
            ms_str = [re.findall('mu_%d.*' % pid,string) for string in trace.varnames if len(re.findall('mu_%d.*' %pid,string))>0]
            ss_str = [re.findall('sigma_%d.*' % pid,string) for string in trace.varnames if len(re.findall('sigma_%d.*'%pid,string))>0]
            ss_str = [[f'{preamb_str}{tmp_str}'] for tmp_arr in ss_str for tmp_str in tmp_arr]

            # Extract posteriors from trace
            m_str = f'{preamb_str}{ms_str[0][0]}'
            ms_post = np.array([np.array([trace[m_str][-num_samps_in:,i,0].mean(), trace[m_str][-num_samps_in:,i,1].mean()])
                                for i in range(my_k)])
            ss_post = np.array([np.array([[trace[s_str[0]][-num_samps_in:,0,0].mean(), trace[s_str[0]][-num_samps_in:,0,1].mean()],
                                          [trace[s_str[0]][-num_samps_in:,1,0].mean(), trace[s_str[0]][-num_samps_in:,1,1].mean()]])
                                for s_str in ss_str])

            if len(data[pid])>0:
                # Plot trained exemplars
                ax.scatter(data_all[pid][0:20, 0], data_all[pid][0:20, 1],s = 20,color='blue',alpha=alpha,marker='o')
                # Plot generated exemplars
                ax.scatter(data_all[pid][20:, 0], data_all[pid][20:, 1],s = 20,color='red',alpha=alpha,marker='x')

            # Plot the ellipses
            plot_ellipse(ax,ms_post,ss_post)
            # Standardize axes
            lim = 2.5
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_ylim(-lim,lim)
            ax.set_xlim(-lim,lim)
            ax.set_title(pid)
            
    # Save the figure for easy future access
    plt.savefig('real_fit%d.pdf' % ki)

## Model comparison
#Convert model and trace into dictionary pairs
dict_pairs = dict(zip(gmm_all,traces))
#Perform WAIC comparison
compare = pm.compare(dict_pairs, ic='WAIC')
#Print comparison
print(compare)
