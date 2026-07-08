import numpy as np
import random

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
    raise RuntimeError("You need to write this part!")

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
    raise RuntimeError("You need to write this part!")

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
    raise RuntimeError("You need to write this part!")


