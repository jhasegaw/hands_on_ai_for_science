# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def center_of_gravity(x):
    '''
    Find the center of gravity of a vector, x.
    If x=[x0,x1,...,xn], then you should return
    c = ( 0*x0 + 1*x1 + 2*x2 + ... + n*xn ) / sum(x)
    where n = len(x)-1.

    Recommended method: use np.arange, np.dot, and np.sum.

    @param:
    x (array): a 1d numpy array

    @result:
    c (scalar): x's center of gravity
    '''
    # The notebook sometimes passes a plain Python list, so convert first.
    # np.asarray leaves it alone if it is already an array.
    x = np.asarray(x)

    # The positions 0, 1, 2, ... n.  Think of x as masses sitting on a ruler,
    # and these as the ruler markings each mass is sitting on.
    positions = np.arange(len(x))

    # np.dot multiplies the two arrays elementwise and adds the results, which
    # is exactly the numerator 0*x0 + 1*x1 + ... + n*xn.  Dividing by the total
    # mass gives the balance point -- the place you could put one finger under
    # the ruler and have it stay level.
    return np.dot(positions, x) / np.sum(x)

def sine_and_cosine(t_start, t_end, t_steps):
    '''
    Create a time axis, and compute its cosine and sine.
    Hint: use np.linspace, np.cos, and np.sin

    @param:
    t_start (scalar): the starting time
    t_end (scalar): the ending time
    t_steps (scalar): length of t, x, and y

    @result:
    t (array of length t_steps): time axis, t_start through t_end inclusive
    x (array of length t_steps): cos(t)
    y (array of length t_steps): sin(t)
    '''
    # np.linspace gives t_steps evenly spaced numbers and, unlike np.arange,
    # it includes the endpoint -- which is what "t_start through t_end
    # inclusive" asks for.
    t = np.linspace(t_start, t_end, t_steps)

    # np.cos and np.sin work on a whole array at once, so there is no loop
    # here: each returns a new array the same length as t.  Note the order of
    # the return values: cosine is x and sine is y, the way you would read a
    # point off the unit circle.
    x = np.cos(t)
    y = np.sin(t)

    return t, x, y
