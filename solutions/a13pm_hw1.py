import numpy as np


def sigma(z):
    '''
    Compute the logistic sigmoid, 1/(1+exp(-z)), for a real number z or numpy array.
    Please use this function as the logistic sigmoid in the gradient function!

    @param:
    z (float) - any real number or numpy array

    @return:
    p (float) - real number between 0 and 1, or numpy array, containing sigma(z)
    '''
    return 1/(1+np.exp(-z))

def initialize(data, dim):
    '''
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])
    
    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.
    
    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    '''
    N = len(set(data))
    theta = np.linspace(0,2*np.pi,N, endpoint=False)
    embedding = {}
    for t in range(len(data)):
        if data[t] not in embedding:
            embedding[data[t]] = np.random.randn(dim)
            embedding[data[t]][0] = np.cos(theta[len(embedding)-1])
            embedding[data[t]][1] = np.sin(theta[len(embedding)-1])
    return embedding

def gradient(embedding, data, t, d=2, k=10):
    '''
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]
    
    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    '''
    vt = embedding[data[t]]
    g = np.zeros(len(vt))
    for c in range(-d,d+1):
        if c != 0 and t+c >= 0 and t+c < len(data):
            vc = embedding[data[t+c]]
            g -= (1-sigma(np.dot(vt,vc)))*vc
            for i in range(k):
                vi = embedding[np.random.choice(data)]
                g += sigma(np.dot(vt,vi))*vi/k
    return g

def sgd(embedding, data, learning_rate, num_iters, d=2, k=10):
    '''
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation
    
    @return:
    embedding - the updated embeddings
    '''
    for i in range(num_iters):
        t = np.random.choice(len(data))
        g = gradient(embedding, data, t, d, k)
        embedding[data[t]] -= learning_rate * g
    return embedding
    

