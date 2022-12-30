from numpy import *
import math
# covariance matrix
sigma = matrix([[4, 2, 1],
                [2, 5, 2],
                [1, 2, 3]])
# mean vector
mu = array([1, 3, 5]) # Transpose added later in result

# locations, in assignment loc = x1, x2, x3
loc = array([2, 2, 2])
# loc = array([1, 4, 3])
# loc = array([1, 1, 5])

def multivariate_gaussian(loc, mu, sigma):
    size = len(loc)
    if size == len(mu) and (size, size) == sigma.shape:
        determinant = linalg.det(sigma) # determinant
        if determinant == 0:
            raise NameError("Error: singular matrix")

        normalize_cons = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(determinant,1.0/2) ) #normalization  #math.exp ->math.pow
        inverse = sigma.I      #inverse matrix
        loc_mu = matrix(loc - mu)    
        result = math.pow(math.e, -0.5 * (loc_mu * inverse * loc_mu.T))  #math.exp ->math.pow
        return normalize_cons * result
    else:
        raise NameError("Error: wrong dimensions")

print (multivariate_gaussian(loc, mu, sigma))


# results: array([2,2,2])^T = 0.0013717986904768756
# results: array([1,4,3])^T = 0.002609033729847758
# results: array([1,1,5])^T = 0.0057241508772283576
