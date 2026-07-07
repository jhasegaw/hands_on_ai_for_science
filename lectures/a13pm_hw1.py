'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise NotImplementedError.  You should replace
each "raise NotImplementedError" line with a line that performs the function specified in the
function's docstring.

'''

import numpy as np

################################
####### NO NEED TO CHANGE ######
################################

def ensure_tensor(x):
    """
    
    #### DO NOT CHANGE ####
    
    Quick helper method to check if something is a tensor
    otherwise convert it. Tensors have a .grad attribute
    unlike Numpy so wwe can check for that!
    """
    if hasattr(x, "grad"):
        return x
    else:
        return Tensor(x, requires_grad=False)
        
class Tensor:

    """

    #### DO NOT CHANGE ####
    
    This is the structure for a tensor! There is no need 
    to change any code here, through the MP we will be defining 
    all of these methods!

    Args:
        - array: numpy array
        - requires_grad: do we need to track gradients on this tensor
        - parents: From what tensor was this new tensor derived from
        - backward_fn: What is the backward method for this operation
    """
    def __init__(self,
                 array,
                 requires_grad=False,
                 parents=(),
                 grad_fn=None,
                 grad_fn_name=None):

        self.array = np.array(array, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.array) if requires_grad else None
        self.parents = parents
        self.grad_fn = grad_fn
        self.grad_fn_name = grad_fn_name

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.grad)

    def backward(self):
        return backward(self)

    @property
    def shape(self):
        return self.array.shape
        
    def __add__(self, other):
        """
        self + other
        """
        return add(ensure_tensor(self), ensure_tensor(other))

    def __radd__(self, other):
        """
        __r(op)__ is a reverse method incase of failure. For example, if we do:

            Tensor + Int -> (Tensor).__add__(Int) This will work as our Tensor class has a __add__ method that can handle this by converting the int
            to a Tensor first and then performing the operation (this will be defined in our add() method later

            Int + Tensor -> (Int).__add__(Tensor) This will fail, as the default Python Int doesnt know how to perform an add operation with a Tensor
            as the Tensor is a class we are making and Python Int has never seen this before. This then triggers Python to try __radd__ which will
            be again Tensor + Int -> (Tensor).__add__(Int) as a backup case to see if this works.

        self + other
        """
        return add(ensure_tensor(self), ensure_tensor(other))
        
    def __sub__(self, other):
        """
        self - other
        """
        return sub(ensure_tensor(self), ensure_tensor(other))

    def __rsub__(self, other):
        """
        Subtraction is an ordered operation. A - B is different from B - A. Therefore if self - other fails, then we reverse it
        so the new self is the old other, and the new other is the old self. When we do our sub() operation though we flip them again
        to keep it consistent!

        self - other
        """
        return sub(ensure_tensor(other), ensure_tensor(self))

    def __mul__(self, other):
        """
        self * other
        """
        return mul(ensure_tensor(self), ensure_tensor(other))

    def __rmul__(self, other):
        """
        self * other
        """
        return mul(ensure_tensor(self), ensure_tensor(other))

    def __truediv__(self, other):
        """
        self / other
        """
        
        return div(ensure_tensor(self), ensure_tensor(other))

    def __rtruediv__(self, other):
        """
        Again an ordered operation, A / B is different from B / A so we flip it again here just like we did in sub
        self / other
        """
        return div(ensure_tensor(other), ensure_tensor(self))

    def __matmul__(self, other):
        """
        A @ B where A is a [P x Q] matrix and B is a [Q x R] to give a final [P x R] matrix
        """
        return matmul(ensure_tensor(self), ensure_tensor(other))

    def __rmatmul__(self, other):
        """
        Matmul is ordered so we flip it around here again!
        """
        return matmul(ensure_tensor(other), ensure_tensor(self))

    def exp(self):
        """
        Element-wise exponentiation 
        
        self.exp()
        """
        return exp(ensure_tensor(self))
        
    def log(self):
        """
        Element wise log (natural log)
        
        self.log()
        """
        return log(ensure_tensor(self))

    def sum(self, axis=None, keepdims=False):
        """
        self.sum(axis=None, keepdims=False) will perform a reduction summation along a dimension (or multiple dimensions) of a tensor. For example

        if our tensor is:

        T = [A B C] -> Shape: (2 x 3)
            [D E F]


        T.sum(axis=0) is a sum along the first dimension (along the column dimension)

            [A+D, B+E, C+F] -> Shape: (3,)

        T.sum(axis=0, keepdims=True) is a sum along the first dimension, but we dont remove that dimension and keep a dummy placeholder 1

            [[A+D, B+E, C+F]] -> Shape: (1,3)

        T.sum(axis=1) is a sum along the second dimension (along the row dimension)

            [A+B+C, D+E+F] -> Shape: (2,)

        T.sum(axis=1, keepdims=True) is a sum along the second dimension but we dont remove that dimension and keep a dummy placeholder 1

            [A+B+C] -> Shape: (2,1)
            [D+E+F]

        T.sum() This is a total sum of all elements in the tensor

            A+B+C+D+E+F -> Shape: () This is just empty, its a total sum to a scaler

        T.sum(keepdims=True)

            [[A+B+C+D+E+F]] -> Shape: (1,1) The 2 dimensional tensor is reduced to a scalar, so two dummy 1 dimensions are added in

        """
        return sum(ensure_tensor(self), axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """
        Identical logic to the sum reduction operation. But this instead does the max along a dimension (or multiple dimensions) of a tensor
        """
        return max(ensure_tensor(self), axis=axis, keepdims=keepdims)

    def __repr__(self):

        """
        Just some pretty printing to make it easier to inspect the Tensors!
        """
        data = self.array

        data_str = np.array2string(
            data,
            separator=" ",
            precision=3,
            floatmode="fixed",
            max_line_width=80
        )

        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("tensor(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        grad_info = ""
        if getattr(self, "requires_grad", False):
            if getattr(self, "grad_fn_name", None) is not None:
                grad_info = f", grad_fn={getattr(self, 'grad_fn_name', None)}"
            else:
                grad_info = ", requires_grad=True"
                
        return f"tensor({data_str}{grad_info})"

def add(a, b):

    """
    
    #### DO NOT CHANGE ####
    
    Y = A + B
    """
    ### Check if this operation requires grad. If neither a nor b requires gradients ###
    ### then we dont need to track the computational graph here ###
    requires_grad = a.requires_grad or b.requires_grad

    ### Compute the forward pass ###
    output = a.array + b.array
    
    ### Define the backward method as a Closure ###
    def backward(grad_out): # grad_out is the upstream gradients

        # grad_out = dL/dy -> our upstream gradients

        ### If a requires grad then we compute it here ###
        if a.requires_grad:

            ### grad_a = dL/dy * 1 = dL/dy
            grad_a = grad_out

            ### Broadcasting accumulation ###
            grad_a = broadcast_grad_accumulate(a, grad_a)

            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

        ### Same logic! ###
        if b.requires_grad:
            grad_b = grad_out
            grad_b = broadcast_grad_accumulate(b, grad_b)
            b.grad += grad_b

    ### Create a new tensor with our attributes 
    return Tensor(
        output, # new tensor is A + B so store the output here
        requires_grad=requires_grad, # Do we need to track grads?
        parents=(a, b), # What tensors were used to create this new one?
        grad_fn=backward if requires_grad else None, # Store the backward method for usage later
        grad_fn_name="<AddBackward>" # Keep a name for visualization
    )
    
########################################
####### START OF THE MP SOLUTIONS ######
########################################

def broadcast_grad_accumulate(tensor, gradient):
    """
    When we do any operation between two tensors, if one is broadcasted over the 
    other we need to do an accumulation of the gradients in the backward pass
    as shown in the notebook. We first implement this here!

    EXAMPLE:

    y = A + B = [1 2 3] + [10 20 30] = [11 22 33]
                [4 5 6]                [14 25 36]

    So we can see that the [10 20 30] is used twice for each row of A (that is broadcasting). 

    In the backward pass gradients come in the shape of the output so we have the upstream grad
    as the derivative of the loss w.r.t y
    
    dL/dy = [d0 d1 d2]
            [d3 d4 d5]

    Grad W.R.T A:
    
    For dL/dA that is easy as dL/dA = dL/dy * dy/dA = dL/dy * 1 = dL/dy = [d0 d1 d2] => (2,3)
                                                                          [d3 d4 d5]

    Notice the shape of our gradient for dL/dA matches the shape of A, so we are good to go!


    Grad W.R.T B:
    
    Because both rows of the original output had contributions from the same 
    broadcasted vector B, the gradient w.r.t. B must accumulate (sum) over the 
    broadcasted dimensions.

    dL/dB = dL/dy * dy/dB = dL/dy * 1 = dL/dy = [d0 d1 d2]
                                                [d3 d4 d5]

    But the shape of our grad (2,3) doesnt match the shape of the original tensor B (3). And
    again this is because B was broadcasted over. So we need to accumulate all those grad
    contributions by summing them together! 

    So actually:

    dL/dB = [[d0 + d3,  d1 + d4,  d2 + d5]] => (1,3)

    Which is the same as just summing along the dimensions that were broadcasted over. In this case, we
    broadcasted over each row of A getting added to the single row of B, so we add together along the 
    columns. 

    GOAL: 

    We want to reproduce this for any arbritary broadcasting. There are three assumption to make this simpler

    Assumption 1:
    
    Tensors have the same shape, and a dimension of 1 indicates that dimension is being broadcasted 

        In the example above, B has a shape of (1,3), but the gradient has a shape of (2,3). With this you know
        that the 1 dimension was broadcasted in B over the 2 dimension in the gradient.

    Assumption 2:

        When we accumulate our gradients, like summing from the (2,3) along the first dimension, we technically
        can just get a tensor of shape (3). But this doesnt match our B tensor which is of shape (1,3). So to 
        deal with this, we will always use keepdims=True so it matches!

        So we would do:

        if grad.shape = (2,3)
        
        dL/dB = grad.sum(axis=0, keepdims=True) => (1,3)

    Assumption 3:

         The gradients can only have the same or more dimensionality (excluding the dummy 1 dimensions). So 
         this is always an accumulation of gradients for repeated uses of tensors like done in broadcasting


    Args:

        tensor: A Tensor object that you have .shape access too
        gradient: A numpy array that you want to accumulate gradients for broadcasting
    
    """

    ### Check Assertion ###
    assert len(tensor.shape) == len(gradient.shape)
    
    ### Sum over dimensions that are size 1 in the original array but not 1 in the gradients (as that indicates
    ### a broadcasted dimension)

    #### WRITE CODE HERE ###
    raise NotImplementedError
    ########################

    ### Make Sure Gradients are the Same Shape as the Tensor ###
    assert gradient.shape == tensor.shape
    
    return gradient

def sub(a, b):

    """
    Y = A - B
    """

    requires_grad = a.requires_grad or b.requires_grad
    
    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output =
    ########################

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:

            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Broadcasting accumulation ###
            grad_a = broadcast_grad_accumulate(a, grad_a)
            
            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

        ### Grad w.r.t B
        if b.requires_grad:

            ### What is dL/dB?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_b = 
            ########################

            ### Broadcasting accumulation ###
            grad_b = broadcast_grad_accumulate(b, grad_b)

            ### Accumulate gradients into the .grad attribute in B
            b.grad += grad_b

    ### Create a new tensor with our attributes 
    return Tensor(
        output, # new tensor is A + B so store the output here
        requires_grad=requires_grad, # Do we need to track grads?
        parents=(a, b), # What tensors were used to create this new one?
        grad_fn=backward if requires_grad else None, # Store the backward method for usage later
        grad_fn_name="<SubBackward>" # Keep a name for visualization
    )

def mul(a, b):

    """
    Y = A*B
    """

    requires_grad = a.requires_grad or b.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output = 
    ######################## 

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:

            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Broadcasting accumulation ###
            grad_a = broadcast_grad_accumulate(a, grad_a)
            
            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

        ### Grad w.r.t B
        if b.requires_grad:

            ### What is dL/dB?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_b = 
            ########################

            ### Broadcasting accumulation ###
            grad_b = broadcast_grad_accumulate(b, grad_b)

            ### Accumulate gradients into the .grad attribute in B
            b.grad += grad_b
            
    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a, b),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<MulBackward>"
    )
    
def div(a, b):

    """
    Y = A / B
    """
    
    requires_grad = a.requires_grad or b.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output =
    ########################

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:
            
            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Broadcasting accumulation ###
            grad_a = broadcast_grad_accumulate(a, grad_a)
            
            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

        ### Grad w.r.t B
        if b.requires_grad:
            
            ### What is dL/dB?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_b = 
            ########################
            
            ### Broadcasting accumulation ###
            grad_b = broadcast_grad_accumulate(b, grad_b)

            ### Accumulate gradients into the .grad attribute in B
            b.grad += grad_b

    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a, b),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<DivBackward>"
    )
    
def matmul(a, b):

    """
    Y = A @ B
    """
    
    requires_grad = a.requires_grad or b.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # out_array = 
    ########################

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:

            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

        ### Grad w.r.t B
        if b.requires_grad:

            ### What is dL/dB?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_b = 
            ########################

            ### Accumulate gradients into the .grad attribute in B
            b.grad += grad_b

    return Tensor(
        out_array,
        requires_grad=requires_grad,
        parents=(a, b),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<MatmulBackward>"
    )

def log(a):

    """
    Y = log(A)
    """

    requires_grad = a.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output =
    ########################

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:

            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a,),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<LogBackward>"
    )

def exp(a):

    """
    Y = exp(A)
    """
    
    requires_grad = a.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output =
    ########################

    ### Define the Backward Function (grad_out is dL/dY) ###
    def backward(grad_out):

        ### Grad w.r.t A
        if a.requires_grad:

            ### What is dL/dA?
            #### WRITE CODE HERE ###
            raise NotImplementedError
            # grad_a = 
            ########################

            ### Accumulate gradients into the .grad attribute in A
            a.grad += grad_a

    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a,),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<ExpBackward>"
    )

def sum(a, axis=None, keepdims=False):

    """
    Reduce sum along a dimension (or multiple dimensions)

    The shape reduction works as follows:
    
        a.shape = (2, 3)
    
        sum(a)              -> shape () -> Grad is also () and then repeated to a (2,3) shape
        sum(a, axis=0)      -> shape (3,) -> Grad is also (3,) and then repeated to a (2,3) shape
        sum(a, axis=1)      -> shape (2,) -> Grad is also (3,) and then repeated to a (2,3) shape
        sum(a, axis=1, keepdims=True) -> shape (2, 1) -> Grad is also (3,) and then repeated to a (2,3) shape
        
    """

    requires_grad = a.requires_grad

    ### Perform the Forward Pass ###
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output = 
    ########################

    ### Keep another copy with keepdims=True for broadcast_to ###
    ### you need the shape of the tensor with keepdims=True for broadcast_to to work, 
    ### if keepdims=True already then not much to do, but if keepdims=False we need that
    ### second copy here to make this simple! 
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # sum_keepdims = 
    ########################
        
    def backward(input_grad):
        
        if a.requires_grad:
            
            #### WRITE CODE HERE ###
            raise NotImplementedError
            ### make sure the input_grad is the same shape as the sum_keepdims first 

            ### Use broadcast_to to repeat gradient over broadcasted dimensions
            
            ### acccumulate grad info 
            # a.grad += 

            #########################

    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a,),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<SumBackward>"
    )

def max(a, axis=None, keepdims=False):

    """
    Very similar to the sum, but now we have max along a dimension (reduction operation again)

    In backprop the derivative of max is 1 at the index that was max and 0 otherwise!
    
    """
    requires_grad = a.requires_grad

    ### Forward pass ###

    #### WRITE CODE HERE ###
    raise NotImplementedError
    # output = 
    ########################

    ### Keep another copy with keepdims=True for broadcast_to ###
    ### you need the shape of the tensor with keepdims=True for broadcast_to to work, 
    ### if keepdims=True already then not much to do, but if keepdims=False we need that
    ### second copy here to make this simple! 
    #### WRITE CODE HERE ###
    raise NotImplementedError
    # max_keepdims = 
    ########################

    def backward(input_grad):

        if a.requires_grad:
            raise NotImplementedError
            #### WRITE CODE HERE ###

            ### make sure the input_grad is the same shape as the sum_keepdims first 

            ### Use broadcast_to to repeat gradient over broadcasted dimensions

            ### Create mask as gradient only flows through max locations 

            ### apply mask to gradient 
            
            ### acccumulate grad info 
            # a.grad += 

            ####################### 

    return Tensor(
        output,
        requires_grad=requires_grad,
        parents=(a,),
        grad_fn=backward if requires_grad else None,
        grad_fn_name="<MaxBackward>"
    )


def backward(x):

    """
    Implementation of the Topological Sort Algorithm
    """
    
    ### Initialize gradients as 1 (base case) in the same shape as our input x
    grad = np.ones_like(x.array)
    x.grad = grad


    ### Initialize the Topological Sort
    topo = []
    visited = set()

    ### Function to compute Topological Sort
    def build_topo(x):

        ### WRITE CODE HERE ###
        raise NotImplementedError
        #######################
    
    ### Build the Topological Sort 
    build_topo(x)
    
    ### Call the backward method for each operation in the reverse order of our sort
    for t in reversed(topo):
        ### if a grad_fn exists for this tensor
        if t.grad_fn is not None:
            ### apply the grad function to this tensors parents with the upstream grad stored in t.grad
            t.grad_fn(t.grad)
    
    ##########################