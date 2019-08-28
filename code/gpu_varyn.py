import pymc3 as pm 
import numpy as np
import pickle
import os.path
import time

#Seed the rng for exact reproducible results
seed = 68492
np.random.seed(seed)

#Load data
with open('data.p','rb') as f:
    data = pickle.load(f)

njobs = 1 #Set to 1 on gpu
exm_basen = len(data[list(data.keys())[0]]) #Number of exemplars per participant (should be 60 by default)
exm_basen_alpha = round(1/3*exm_basen) 
exm_basen_beta = exm_basen-exm_basen_alpha    
n_ppt = len(data) #Sample size
ppt_list_base = list(data.keys())
ppt_jitter = .05 #Add jitter

data_original = data.copy()
times = dict()
timesfn = 'times_varyn.p'
varyK = [1,2,3,5] #Number of components
niter = 2000


factor_list = [1,2,5,10,50]#[.05,.1,.5,1,2,5,10,50,100] 

#Define function for building model -- essentially same as that in realdata.py
def build_model(data,K):
    n_ppt = len(data)
    print('Building model with n=%d,K=%d' % (n_ppt,K))
    with pm.Model() as gmm:
        #Prior
        if K>1:
            p = pm.Dirichlet('p', a=pm.floatX(np.array([1.]*K)), testval=pm.floatX(np.ones(K)/K))
        mus_p = [pm.MvNormal('mu_%s' % pid,
                             mu=pm.floatX(np.zeros(2)),
                             tau=pm.floatX(0.1 * np.eye(2)),
                             shape=(K,2))
                 for pi,pid in enumerate(data.keys())]

        packed_L = [[pm.LKJCholeskyCov('packed_L_%s_%d' % (pid,i),
                                       n=2,
                                       eta=pm.floatX(2.),
                                       sd_dist=pm.HalfCauchy.dist(.01))
                    for i in range(K)]
                    for pi,pid in enumerate(data.keys())]
        L = [[pm.expand_packed_triangular(2, packed_L[pi][i])
            for i in range(K)]
            for pi,pid in enumerate(data.keys())]

        sigma = [[pm.Deterministic('sigma_%s_%d' % (pid,i) ,L[pi][i].dot(L[pi][i].T))
                for i in range(K)]
                for pi,pid in enumerate(data.keys())]

        if K>1:
            mvnl = [[pm.MvNormal.dist(mu=mus_p[pi][i],chol=L[pi][i])
                    for i in range(K)]
                    for pi in range(n_ppt)]
            Y_obs = [pm.Mixture('Y_obs_%s' % pid,w=p, comp_dists=mvnl[pi],observed=data[pid])
                     for pi,pid in enumerate(data.keys())]
        else:
            Y_obs = [pm.MvNormal('Y_obs_%s' % pid, mu=mus_p[pi][0],chol=L[pi][0],observed=data[pid])
                     for pi,pid in enumerate(data.keys())]

    return gmm


def sample_model(gmm,K,N):
    with gmm:
        print('Sampling from Model with {} component(s), n={}, d={}.'.format(K,N,2))
        step = pm.NUTS()
        #Start the sampler! For this larger dataset, it might take awhile -- allocate a few hours.
        trace = pm.sample(niter, step=step,chains=4,njobs=njobs)
        #trace = pm.sample(5, step=step,chains=1,njobs=njobs,tune=0,progressbar=True)
    return trace

for K in varyK:
    times[K] = dict()
    for ftr in factor_list:
        print('Running simulations with data size multiplied by a factor of {}'.format(ftr))
        ppt_factor = ftr #Clone the data by participants by this much

        #Clone data by bootstrap method
        data = data_original.copy()
        ppt_toclone = ppt_factor*len(ppt_list_base) - len(ppt_list_base)
        for pc in range(ppt_toclone):
            #Sample one ppt for alphas
            ppt_alpha = np.random.choice(ppt_list_base)
            alphas = data[ppt_alpha][:exm_basen_alpha,:]     
            #Sample again for betas
            ppt_beta = np.random.choice(ppt_list_base)
            betas = data[ppt_beta][exm_basen_alpha:exm_basen_beta,:]
            #Stack together as new ppt
            pc_data = np.concatenate([alphas,betas],axis=0)
            pc_name = str(pc)+'c' #'c' for clone
            #Add jitter
            pc_data += np.random.sample(pc_data.shape) * ppt_jitter
            #Add to data dictionary
            data[pc_name] = pc_data

        n_ppt = len(data)
        times[K][n_ppt] = dict()    


        #Start timing
        startbd = time.time()
        # Pass model to gmm
        gmm = build_model(data,K)
        #End timing
        endbd = time.time()
        times[K][n_ppt]['build'] = endbd-startbd
        print('Done building. Took {} seconds.'.format(endbd-startbd))

        #Start timing for sampling
        startsm = time.time()
        #Sample
        sample_model(gmm,K,n_ppt)
        #End timing for sampling
        endsm = time.time()    
        times[K][n_ppt]['sample'] = endsm-startsm
        print('Done sampling. Took {} seconds.'.format(endsm-startsm))

        #Load current time if it exists
        if os.path.isfile(timesfn): 
            with open(timesfn,'rb') as f:
                times_old = pickle.load(f)
            if not K in times_old.keys():
                times_old[K] = dict()
            #overwrite old times with current run
            times_old[K][n_ppt] = times[K][n_ppt]
            times = times_old
        #Save as pickle
        with open(timesfn,'wb') as f:
            pickle.dump(times,f)

