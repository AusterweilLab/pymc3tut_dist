import pymc3 as pm 
import numpy as np
import pickle
import os.path
import time

##Uncomment the two lines below if you're getting weird nesting bracket error
#import theano
#theano.config.gcc.cxxflags = "-fbracket-depth=1024"

#Seed the rng for exact reproducible results
seed = 68492
np.random.seed(seed)

njobs = 1 #Set to 1 on gpu

Kbase = 1 #Number of components
Nbase = 100#1000
dbase = 2

#Load data
datafn = 'gpu_data_varyd.p'.format(Nbase)
with open(datafn,'rb') as f:
    dataset = pickle.load(f)

varykeys = ['varyd']
times = dict()
times['varyd'] = dict()
timesfn = 'times_varyd.p'

niter = 1000

varlist = dict()
varlist['varyd'] = [100,200,400,500,600,800,1000]
varyK = [1,2,3,5]

## Model building function
def build_model(data,K):
    N = data.shape[0]
    d = data.shape[1]
    print('Building model with n=%d, d=%d, k=%d' % (N,d,K))
    with pm.Model() as gmm:
        #Prior over component weights
        if K>1:
            p = pm.Dirichlet('p', a=np.array([1.]*K))
            
        #Prior over component means
        mus = [pm.MvNormal('mu_%d' % i,
                            mu=pm.floatX(np.zeros(d)),
                            tau=pm.floatX(0.1 * np.eye(d)),
                            shape=(d,))
                            #testval = pm.floatX(np.ones(d)))
               for i in range(K)]
        #Cholesky decomposed LKJ prior over component covariance matrices
        packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i,
                                      n=d,
                                      eta=2.,
                                      sd_dist=pm.HalfCauchy.dist(1))
                                      #testval = pm.floatX(np.ones(int(d*(d-1)/2+d))))
                    for i in range(K)]
        #Unpack packed_L into full array
        L = [pm.expand_packed_triangular(d, packed_L[i])
             for i in range(K)]
        #Convert L to sigma and tau for convenience
        sigma = [pm.Deterministic('sigma_%d' % i ,L[i].dot(L[i].T))
                 for i in range(K)]
        tau = [pm.Deterministic('tau_%d' % i,matrix_inverse(sigma[i]))
               for i in range(K)]        
        
        #Specify the likelihood
        if K>1:
            mvnl = [pm.MvNormal.dist(mu=mus[i],chol=L[i])  
                    for i in range(K)]    
            Y_obs = pm.Mixture('Y_obs',w=p, comp_dists=mvnl,observed=data)  
        else:
            Y_obs = pm.MvNormal('Y_obs',mu=mus[0],chol=L[0],observed=data)

    return gmm

## Sampling function
def sample_model(gmm,K,N,d):
    with gmm:
        print('Sampling from Model with {} component(s), n={}, d={}.'.format(K,N,d))
        step = pm.NUTS()
        #Start the sampler! For this larger dataset, it might take awhile -- allocate a few hours.
        trace = pm.sample(niter, step=step,chains=4,njobs=njobs)
        #trace = pm.sample(5, step=step,chains=1,njobs=njobs,tune=0,progressbar=True)
    return trace

## Loop over varyd
for varykey in varykeys:
    for K in varyK:
        print('Varying {} with {} components'.format(varykey[-1],K))
        times[varykey][K] = dict()
        for varyval in varlist[varykey]:
            #Extract it for easier processing
            data = dataset[varykey][varyval]        
            N = Nbase
            d = varyval
            varykey = 'varyd'            
            times[varykey][K][varyval] = dict()    

            startbd = time.time()
            # Pass model to gmm
            gmm = build_model(data,K)
            endbd = time.time()
            times[varykey][K][varyval]['build'] = endbd-startbd
            print('Done building. Took {} seconds.'.format(endbd-startbd))

            startsm = time.time()
            sample_model(gmm,K,N,d)
            endsm = time.time()
            times[varykey][K][varyval]['sample'] = endsm-startsm
            print('Done sampling. Took {} seconds.'.format(endsm-startsm))

            #Load current times if it exists
            if os.path.isfile(timesfn): 
                with open(timesfn,'rb') as f:
                    times_old = pickle.load(f)
                if not K in times_old[varykey].keys():
                    times_old[varykey][K] = dict()
                #overwrite old times with current run
                times_old[varykey][K][varyval] = times[varykey][K][varyval]
                times = times_old
            #Save as pickle
            with open(timesfn,'wb') as f:
                pickle.dump(times,f)



