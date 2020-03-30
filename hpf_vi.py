import numpy as np 
from scipy.special import digamma
import time
from sklearn.metrics import mean_squared_error

class hpf_vi():
    def __init__(self, a = 0.3, c = 0.3, a1 = 0.3, b1 = 1, c1 = 0.3, d1 = 1, K = 10):
        '''
        Initialization of the parameter matrices used in the CAVI algorithm.
        The user can modify the hyperparameters and the dimension of latent attributes and preferences K.

        Parameters:
        ----------
            - a : float
              shape parameter for the Gamma(a, activity_u) prior
              for user preferences.
            - c : float
              shape parameter for the Gamma(c, popularity_i) prior
              for item attributes.
            - a1, b1 : floats
              parameters of the Gamma(a1, a1/b1) prior for user activity.
            - c_1, d_1 : floats
              parameters of the Gamma(c1, c1/d1) prior for item popularity.
            - K : int
              dimensionality of latent attributes and preferences.
        '''
        self.a, self.c, self.a1, self.b1, self.c1, self.d1, self.K = a, c, a1, b1, c1, d1, K

    def fit(self, train, iterations):
        '''
        Fit the Hierarchical Poisson Factorization model via Coordinate Ascent Variational Algorithm (CAVI)
        and generates the corresponding observation matrix based on the variational parameters.
        
        Parameters:
        ----------
            - train : numpy.array
              UxI array with U = users and I = items.
            - iterations: int
              number of desired training epochs.
        '''
        # Dataset dimensions
        self.U, self.I = train.shape

        self.initialize()
        

        import time
        tic = time.clock()
        for iter in range(iterations):
            for u, i in zip(train.nonzero()[0], train.nonzero()[1]):
                self.phi[u,i] = [np.exp(digamma(self.gamma_shp[u,k]) - np.log(self.gamma_rte[u,k])\
                + digamma(self.lambda_shp[i,k]) - np.log(self.lambda_rte[i,k])) for k in range(self.K)]
                self.phi[u,i] = self.phi[u,i] / np.sum(self.phi[u,i])
            
            for u in range(self.U):
                self.gamma_shp[u] = [self.a + np.sum(train[u]*self.phi[u,:,k]) for k in range(self.K)]
                self.gamma_rte[u] = [self.k_shp/self.k_rte[u] + np.sum(self.lambda_shp[:,k]/self.lambda_rte[:,k]) for k in range(self.K)]
                self.k_rte[u] = self.a1/self.b1 + np.sum(self.gamma_shp[u]/self.gamma_rte[u])

            for i in range(self.I):
                self.lambda_shp[i] = [self.c + np.sum(train[:,i]*self.phi[:,i,k]) for k in range(self.K)]
                self.lambda_rte[i] = [self.tau_shp/self.tau_rte[i] + np.sum(self.gamma_shp[:,k]/self.gamma_rte[:,k]) for k in range(self.K)]
                self.tau_rte[i] = self.c1/self.d1 + np.sum(self.lambda_shp[i]/self.lambda_rte[i])

        # Building user preferences and item attributes
        self.theta = self.gamma_shp/self.gamma_rte
        self.beta = self.lambda_shp/self.lambda_rte

        # Generating observations y
        self.predicted = np.dot(self.theta, self.beta.T)

        toc = time.clock()
        time = toc - tic
        print(f"Completed in {time} seconds")

    def mean_squared_error(self):
        '''
        Assess the convergence of the algorithm.
        Please note that CAVI might find local optima. Moreover, this is only to qualitatively assess if our algorithm has somewhat reached a plausible result.
        '''
        from sklearn.metrics import mean_squared_error 
        self.mse_error = mean_squared_error(train.flatten(), self.predicted.flatten())
    
    def recommend(self, test, t = 0.3):
        '''
        Using the fitted algorithm to make recommendations to users of our dataset.
                
        Parameters:
        ----------
            - t : float
              Delta-threshold for activating recommendations
        '''
        for u in range(self.U):
            recomm = []
            for i in range(self.I):
                if self.predicted[u,i] > t:
                    recomm.append(i)
            if [i > 0 for i in recomm]:
                print(f"User {u} may also like these items: {recomm}")

    def initialize(self):
        self.gamma_shp = np.random.uniform(0,1, size = (self.U, self.K)) + self.a
        self.gamma_rte = np.repeat(self.a/self.b1,self.K) + np.random.uniform(0,1, size = (self.U, self.K))
        self.k_rte = self.a1/self.b1 + np.random.uniform(0,1, self.U)
        self.k_shp = self.a1 + self.K*self.a

        self.lambda_shp = np.random.uniform(0,1, size = (self.I, self.K)) + self.c
        self.lambda_rte = np.repeat(self.c/self.d1,self.K) + np.random.uniform(0,1, size = (self.I, self.K))
        self.tau_rte = self.c1/self.d1 + np.random.uniform(0,1, self.I)
        self.tau_shp = self.c1 + self.K*self.c
        # Note that the parameters tau_shp and k_shp are not updated in the algorithm, so they are declared here.

        self.phi = np.zeros(shape=[self.U, self.I, self.K])
