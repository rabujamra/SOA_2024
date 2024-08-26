import numpy as np

# Cost Function
def calc_cost(T, n_samples):
    # Define cost tiers
    cost_low = 1
    cost_medium = 5
    cost_high = 10

    # Assign costs based on treatment type
    C = np.where(T == 0, np.random.normal(loc=cost_low, scale=2, size=n_samples), 
                 np.where(T == 1, np.random.normal(loc=cost_medium, scale=2, size=n_samples), 
                      np.random.normal(loc=cost_high, scale=2, size=n_samples)))
    C = np.maximum(C, 0.1)  # Ensure costs are positive

    return C
    
# Reward Functions
def R0_k(X,A,c):
    n_samples = len(A)
    mu = (X[:,0]+X[:,1])*(A==0) + (X[:,0]-X[:,1])*(A==1) + (X[:,1]-X[:,0])*(A==2)
    return np.random.normal(loc = mu, scale = .1)
    
def R1_k(X,A,c):
  n_samples = len(A)
  mu = 1 + 2*X[:,0] + X[:,1] + 0.5*X[:,2]
  epsilon = np.random.multivariate_normal(mean = [0]*n_samples, cov = np.eye(n_samples))  
  #epsilon = np.random.gamma(shape=1, scale=1, size=n_samples)
  func = c-X[:,0]-X[:,1]
  return mu + .442*func*(2*A-1) + epsilon

def R2_k(X, A, c, p_outlier = 0):
  n_samples = len(A)
  mu = 1 + 2*X[:,0] + X[:,1] + 0.5*X[:,2]
  epsilon = np.random.multivariate_normal(mean = [0]*n_samples, cov = np.eye(n_samples))
  #epsilon = np.random.gamma(shape=1, scale=1, size=n_samples) 
  func = X[:,1] - 0.25*X[:,0]**2 - 0.5
  return mu + .442*func*(2*A-1) + epsilon

def R3_k(X, A, p_outlier = 0):
  n_samples = len(A)
  outliers = np.random.binomial(n=1, p =p_outlier, size = n_samples)
  mu = 1 + 2*X[:,0] + X[:,1] + 0.5*X[:,2]
  epsilon = np.random.multivariate_normal(mean = [0]*n_samples, cov = np.eye(n_samples))
  #epsilon = np.random.gamma(shape=1, scale=1, size=n_samples) 
  func = (.5 - X[:,0]**2 - X[:,1]**2)*(X[:,0]**2 + X[:,1]**2 - 0.3)
  return mu + func*(2*A-1) + epsilon

def R4_k(X, A, p_outlier = 0):
  n_samples = len(A)
  outliers = np.random.binomial(n=1, p =p_outlier, size = n_samples)
  mu = 1 + 2*X[:,0] + X[:,1] + 0.5*X[:,2]
  epsilon = np.random.multivariate_normal(mean = [0]*n_samples, cov = np.eye(n_samples))
  #epsilon = np.random.gamma(shape=1, scale=1, size=n_samples) 
  func = 1 - X[:,0]**3 + np.exp(X[:,2]**2 + X[:,4]) + 0.6*X[:,5] - (X[:,6] + X[:,7])**2
  return mu + func*(2*A-1) + epsilon


# Optimal Treatments 
def optimal0_k(X,c):
  return (1-(X[:,0] >0)*(X[:,1]>0))*((X[:,0] > X[:,1]) + 2*(X[:,1] > X[:,0]))
    
def optimal1_k(X,c):
  return (c-X[:,0]-X[:,1] > 0)

def optimal2_k(X,c):
  return (X[:,1] - 0.25*X[:,0]**2 - 0.5 > 0)

def optimal3_k(X,c):
  return (0.7 - X[:,0]**2 - X[:,1]**2)*(X[:,0]**2 + X[:,1]**2 - 0.1) > 0

def optimal4_k(X,c):
  return (1 - X[:,0]**3 + np.exp(X[:,2]**2 + X[:,4]) + 0.6*X[:,5] - (X[:,6] + X[:,7])**2 > 0)
