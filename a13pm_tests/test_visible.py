import unittest
import submitted
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight

class TestStep(unittest.TestCase):

    ### TEST BROADCAST_GRAD_ACCUMULATE ###
    @weight(5)
    def test_broadcast_no_broadcasting(self):
        """
        Testcase where we have an op with no broadcasting (input/grad are same shape)
        """

        input_shape = (4,12)
        grad_shape = (4,12)

        ### The data we will use ###
        tensor = np.ones(shape=input_shape)
        grad = np.ones(shape=grad_shape)
        
        ### Because input and grad are the same shape, no accumulation ###
        ref = np.ones(shape=input_shape)

        ### From Submitted File ###
        hyp = submitted.broadcast_grad_accumulate(tensor, grad)

        ### Check same ###
        self.assertEqual(hyp.shape, input_shape)
        np.testing.assert_array_equal(ref, hyp)
    
    @weight(5)
    def test_broadcast_1d(self):
        """
        Testcase where we have an op with 1d broadcasting input: (4,1), grad: (4,12)
        """

        input_shape = (4,1)
        grad_shape = (4,12)

        ### The data we will use ###
        tensor = np.ones(shape=input_shape)
        grad = np.ones(shape=grad_shape)
        
        ### Because input and grad are the same shape, no accumulation ###
        ref = np.ones(shape=input_shape) * 12

        ### From Submitted File ###
        hyp = submitted.broadcast_grad_accumulate(tensor, grad)
        
        ### Check same ###
        self.assertEqual(hyp.shape, input_shape)
        np.testing.assert_array_equal(ref, hyp)

    @weight(5)
    def test_broadcast_2d(self):
        """
        Testcase where we have an op with 2d broadcasting input: (1,1), grad: (4,12)
        """

        input_shape = (1,1)
        grad_shape = (4,12)

        ### The data we will use ###
        tensor = np.ones(shape=input_shape)
        grad = np.ones(shape=grad_shape)
        
        ### Because input and grad are the same shape, no accumulation ###
        ref = np.ones(shape=(input_shape)) * 48

        ### From Submitted File ###
        hyp = submitted.broadcast_grad_accumulate(tensor, grad)
        
        ### Check same ###
        self.assertEqual(hyp.shape, input_shape)
        np.testing.assert_array_equal(ref, hyp)

    @weight(5)
    def test_broadcast_arbritary(self):
        """
        Testcase where we have an op with 1d broadcasting input: (4,1,6,1,1), grad: (4,12,6,2,6)
        """

        input_shape = (4,1,6,1,1)
        grad_shape = (4,12,6,2,6)

        ### The data we will use ###
        tensor = np.ones(shape=input_shape)
        grad = np.ones(shape=grad_shape)
        
        ### Because input and grad are the same shape, no accumulation ###
        ref = np.ones(shape=(input_shape)) * 144

        ### From Submitted File ###
        hyp = submitted.broadcast_grad_accumulate(tensor, grad)
 
        ### Check same ###
        self.assertEqual(hyp.shape, input_shape)
        np.testing.assert_array_equal(ref, hyp)

    ### TEST SUB ###
    @weight(3)
    def test_subtraction_no_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,12))*2, requires_grad=True)

        ### compute output ###
        output = A - B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(A.array) * -1)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, np.ones_like(A.array))
        np.testing.assert_array_equal(B.grad, np.ones_like(B.array) * -1)

    @weight(3)
    def test_subtraction_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,1))*2, requires_grad=True)

        ### compute output ###
        output = A - B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(A.array) * -1)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, np.ones_like(A.array))
        np.testing.assert_array_equal(B.grad, np.ones_like(B.array) * -12)

    ### TEST Mul ###
    @weight(3)
    def test_multiplication_no_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,12))*2, requires_grad=True)

        ### compute output ###
        output = A * B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(A.array) * 2)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, np.ones_like(A.array) * 2)
        np.testing.assert_array_equal(B.grad, np.ones_like(B.array))

    @weight(3)
    def test_multiplication_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,1)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,12))*2, requires_grad=True)

        ### compute output ###
        output = A * B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(B.array) * 2)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, np.ones_like(A.array) * 2 * 12)
        np.testing.assert_array_equal(B.grad, np.ones_like(B.array))

    ### TEST Div ###
    @weight(3)
    def test_division_no_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,12))*2, requires_grad=True)

        ### compute output ###
        output = A / B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(A.array) * 0.5)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, 1 / B.array)
        np.testing.assert_array_equal(B.grad, (-A.array / (B.array ** 2)))

    @weight(3)
    def test_division_broadcast(self):

        A = submitted.Tensor(np.ones(shape=(4,1)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(4,12))*2, requires_grad=True)

        ### compute output ###
        output = A / B
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.ones_like(B.array) * 0.5)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, (1 / B.array).sum(axis=-1, keepdims=True))
        np.testing.assert_array_equal(B.grad, (-A.array / (B.array ** 2)))

    ### TEST Matmul ###
    @weight(3)
    def test_matmul(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)
        B = submitted.Tensor(np.ones(shape=(12,8))*2, requires_grad=True)

        ### compute output ###
        output = A @ B
        upstream_grad = np.ones_like(output.array) * 0.2

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, A.array @ B.array)
        self.assertEqual(A.grad.shape, A.array.shape)
        self.assertEqual(B.grad.shape, B.array.shape)
        np.testing.assert_array_equal(A.grad, upstream_grad @ B.array.T)
        np.testing.assert_array_equal(B.grad, A.array.T @ upstream_grad)

    ### TEST Log ###
    @weight(3)
    def test_log(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)

        ### compute output ###
        output = A.log()
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.log(A.array))
        self.assertEqual(A.grad.shape, A.array.shape)
        np.testing.assert_array_equal(A.grad, (1 / A.array) * upstream_grad)
    
    ### TEST Exp ###
    @weight(3)
    def test_exp(self):

        A = submitted.Tensor(np.ones(shape=(4,12)), requires_grad=True)

        ### compute output ###
        output = A.exp()
        upstream_grad = np.ones_like(output.array)

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(output.array, np.exp(A.array))
        self.assertEqual(A.grad.shape, A.array.shape)
        np.testing.assert_array_equal(A.grad, output.array * upstream_grad)

    ### Test Reduce Sum ###
    @weight(3)
    def test_sum_reduction_1d_no_keepdims(self):

        A = submitted.Tensor([[1,1,1],[1,1,1]], requires_grad=True)

        ### Compute Output ###
        output = A.sum(axis=1, keepdims=False)
        upstream_grad = np.array([1,2])

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[1,1,1],[2,2,2]]))

    @weight(4)
    def test_sum_reduction_1d_keepdims(self):

        A = submitted.Tensor([[1,1,1],[1,1,1]], requires_grad=True)

        ### Compute Output ###
        output = A.sum(axis=1, keepdims=True)
        upstream_grad = np.array([[1],[2]])

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[1,1,1],[2,2,2]]))
    
    @weight(4)
    def test_sum_reduction_2d_no_keepdims(self):

        A = submitted.Tensor([[1,1,1],[1,1,1]], requires_grad=True)

        ### Compute Output ###
        output = A.sum(keepdims=False)
        upstream_grad = np.array([2])

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[2,2,2],[2,2,2]]))
    
    @weight(4)
    def test_sum_reduction_2d_keepdims(self):

        A = submitted.Tensor([[1,1,1],[1,1,1]], requires_grad=True)

        ### Compute Output ###
        output = A.sum(keepdims=True)
        upstream_grad = np.array([[2]])

        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[2,2,2],[2,2,2]]))

    ### Test Reduce Max ###
    @weight(4)
    def test_max_reduction_1d_no_keepdims(self):

        A = submitted.Tensor([[1,3,2],[4,2,7]], requires_grad=True)

        ### Compute Output ###
        output = A.max(axis=1, keepdims=False)
        upstream_grad = np.array([1,2])
      
        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[0,1,0],[0,0,2]]))

    @weight(4)
    def test_max_reduction_1d_keepdims(self):

        A = submitted.Tensor([[1,3,2],[4,2,7]], requires_grad=True)

        ### Compute Output ###
        output = A.max(axis=1, keepdims=True)
        upstream_grad = np.array([[1],[2]])
      
        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[0,1,0],[0,0,2]]))
    
    @weight(4)
    def test_max_reduction_2d_no_keepdims(self):

        A = submitted.Tensor([[1,3,2],[4,2,7]], requires_grad=True)

        ### Compute Output ###
        output = A.max(keepdims=False)
        upstream_grad = np.array([2])
      
        ### compute gradients ###
        output.grad_fn(upstream_grad)
  
        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[0,0,0],[0,0,2]]))
    
    @weight(4)
    def test_max_reduction_2d_keepdims(self):

        A = submitted.Tensor([[1,3,2],[4,2,7]], requires_grad=True)

        ### Compute Output ###
        output = A.max(keepdims=True)
        upstream_grad = np.array([[2]])
      
        ### compute gradients ###
        output.grad_fn(upstream_grad)

        ### check outputs and grads ###
        np.testing.assert_array_equal(A.grad, np.array([[0,0,0],[0,0,2]]))

    ### Test Backpropagation ###
    @weight(8)
    def test_backprop_sigmoid(self):

        A = submitted.Tensor([[1,3,2],[4,2,7]], requires_grad=True)

        def sigmoid(x):
            return 1 / (1 + (-1 * x).exp())
        
        ### Perform sigmoid ###
        output = sigmoid(A)

        ### Trigger Backward ###
        output.backward()

        np.testing.assert_allclose(A.grad, output.array * (1 - output.array), rtol=1e-4)

    @weight(8)
    def test_backprop_square(self):

        A = submitted.Tensor([[1.0, -2.0, 3.0],
                              [4.0, -1.0, 2.0]], requires_grad=True)

        out = A * A
        out.backward()

        expected_grad = 2 * A.array
        np.testing.assert_allclose(A.grad, expected_grad, rtol=1e-4)

    @weight(8)
    def test_backprop_cubic(self):

        A = submitted.Tensor([[1.0, -2.0],
                            [3.0, 0.5]], requires_grad=True)

        out = A * A * A
        out.backward()

        expected_grad = 3 * (A.array ** 2)
        np.testing.assert_allclose(A.grad, expected_grad, rtol=1e-4)

    @weight(8)
    def test_backprop_quadratic_polynomial(self):

        A = submitted.Tensor([[1.0, 2.0],
                             [-1.0, 3.0]], requires_grad=True)

        out = 3 * (A * A) + 2 * A + 1
        out.backward()

        expected_grad = 6 * A.array + 2
        np.testing.assert_allclose(A.grad, expected_grad, rtol=1e-4)

    @weight(8)
    def test_backprop_logsumexp_vector(self):
        
        """
        Derivative of logsumexp is just softmax!
        """
        A = submitted.Tensor([1.0, 2.0, 3.0], requires_grad=True)

        out = (A.exp().sum()).log()
        out.backward()

        expA = np.exp(A.array)
        expected_grad = expA / expA.sum()

        np.testing.assert_allclose(A.grad, expected_grad, rtol=1e-4)

    @weight(8)
    def test_backprop_logsumexp_axis(self):

        A = submitted.Tensor([[1.0, 2.0, 3.0],
                             [0.0, 1.0, 2.0]], requires_grad=True)

        out = (A.exp().sum(axis=1)).log().sum()
        out.backward()

        expA = np.exp(A.array)
        softmax = expA / expA.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(A.grad, softmax, rtol=1e-4)