import numpy as np
import random


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
    In order to make debugging easier, however, we will initialize the words so that
    they are spaced uniformly around the outsides of dim/2 circles.  Specifically, if
    d is any even number less than dim, then the d'th element of the embedding for the 
    n'th word should be np.cos(n*d*np.pi/N), and the (d+1)st element should be
    np.sin(n*d*np.pi/N).
    
    @param:
    data (list) - list of word tokens in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding = a dictionary, embedding[w] = numpy array of length==dim 
    '''
    # raise RuntimeError("You need to write this part!")
    embedding = { w:np.zeros(dim) for w in data }
    N = len(embedding)
    for n,w in enumerate(embedding.keys()):
        embedding[w] = np.zeros(dim)
        for d in range(0,dim,2):
            embedding[w][d] = np.cos(n*d*np.pi/N)
            embedding[w][d+1] = np.sin(n*d*np.pi/N)
    return embedding

def negativegradient(embedding, data, t):
    '''
    Calculate direction of the update for the embedding of word data[t]
    using skipgram noise contrastive embedding, using +/- 3 words 
    as context words, and with the "noise" estimated using all words 
    in the dictionary.
    
    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient

    @return:
    d (numpy array) - update direction for the embedding of word data[t]
    '''
    #raise RuntimeError("You need to write this part!")
    w = embedding[data[t]]
    d = np.zeros(len(w))
    for c in range(max(0,t-3),min(len(data),t+4)):
        if c != 0:
            v = embedding[data[c]]
            s = 1/(1+np.exp(-np.dot(w,v)))
            d += (1-s)*v / 6
    for v in embedding.values():
        s = 1/(1+np.exp(-np.dot(w,v)))
        d -= s*v/9
    return d

def sgd(embedding, data, learning_rate, num_iters):
    '''
    Perform num_iters steps of stochastic gradient descent.
    In each iteration, choose a word at random, compute 
    its update direction using negativegradient,
    then move in that direction with a step size of learning_rate.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform

    @return:
    embedding - the updated embeddings
    '''
    #raise RuntimeError("You need to write this part!")
    for iter in range(num_iters):
        t = random.randrange(len(data))
        d = negativegradient(embedding, data, t)
        embedding[data[t]] += learning_rate*d
    return embedding

