import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from theano.tensor.nlinalg import matrix_inverse
from ellipse import plot_ellipse

# Seed the rng for exact reproducible results
seed = 68492
np.random.seed(seed)

## Generate some data for three groups.
# Means, variances, covariances, and proportions
mus = np.array([[-4,2],[0,1],[6,-2]]) 
variance1   = [1,.4,1.5]
variance2   = [1,.8,5  ]
covariances = [.5,0, -1]
ps = np.array([0.2, 0.5, 0.3])
D = mus[0].shape[0]
# Total amount of data
N = 1000
# Number of groups
K = 3
# Form covariance matrix for each group
sigmas = [np.array([[variance1[i],covariances[i]],[covariances[i],variance2[i]]]) for i in range(K)]
# Form group assignments
zs = np.array([np.random.multinomial(1, ps) for _ in range(N)]).T
xs = [z[:, np.newaxis] * np.random.multivariate_normal(m, s, size=N)
      for z, m, s in zip(zs, mus, sigmas)]
# Stack data into single array
data = np.sum(np.dstack(xs), axis=2)

## Plot them nicely
# Prepare subplots
fig, ax = plt.subplots(figsize=(8, 6))
# First, scatter
plt.scatter(data[:, 0], data[:, 1], c='g', alpha=0.5)
plt.scatter(mus[0, 0], mus[0, 1], c='r', s=100)
plt.scatter(mus[1, 0], mus[1, 1], c='b', s=100)
plt.scatter(mus[2, 0], mus[2, 1], c='y', s=100)
# Then, ellipses
plot_ellipse(ax,mus,sigmas)
ax.axis('equal')
plt.show()

## Build model and sample
# Number of iterations for sampler
draws = 2000
# Prepare lists of starting points for mu to prevent label-switching problem
testvals = [[-2,-2],[0,0],[2,2]]

# Model structure
with pm.Model() as mvgmm:
    # Prior over component weights
    p = pm.Dirichlet('p', a=np.array([1.]*K))

    # Prior over component means
    mus = [pm.MvNormal('mu_%d' % i,
                        mu=pm.floatX(np.zeros(D)),
                        tau=pm.floatX(0.1 * np.eye(D)),
                        shape=(D,),
                        testval=pm.floatX(testvals[i]))
           for i in range(K)]

    # Cholesky decomposed LKJ prior over component covariance matrices
    packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i,
                                  n=D,
                                  eta=2.,
                                  sd_dist=pm.HalfCauchy.dist(1))
                for i in range(K)]

    # Unpack packed_L into full array
    L = [pm.expand_packed_triangular(D, packed_L[i])
         for i in range(K)]

    # Convert L to sigma and tau for convenience
    sigma = [pm.Deterministic('sigma_%d' % i ,L[i].dot(L[i].T))
             for i in range(K)]
    tau = [pm.Deterministic('tau_%d' % i,matrix_inverse(sigma[i]))
           for i in range(K)]

    # Specify the likelihood
    mvnl = [pm.MvNormal.dist(mu=mus[i],chol=L[i])  
           for i in range(K)]    
    Y_obs = pm.Mixture('Y_obs',w=p, comp_dists=mvnl,observed=data)  

    # Start the sampler!
    trace = pm.sample(draws, step=pm.NUTS(), chains=4)
    
## Plot traces
pm.traceplot(trace, varnames=['p', 'mu_0','mu_1','mu_2','tau_0','tau_1','tau_2','sigma_0','sigma_1','sigma_2'])
plt.show()
