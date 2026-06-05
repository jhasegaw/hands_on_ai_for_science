
import unittest, submitted, json
import numpy as np
from gradescope_utils.autograder_utils.decorators import (
    weight,
    visibility,
    partial_credit,
)

class TestCase(unittest.TestCase):
    def setUp(self): 
        self.data = '''% % % 
            aa bb cc aa bb cc aa bb cc aa bb cc
            % % %
            ee ff gg ee ff gg ee ff gg ee ff gg
            % % %'''.split()
        
        self.dataset = np.load("data/sst_visible.npy", allow_pickle=True)
        with open('solution.json') as f:
            self.solution = json.load(f)
    
    @weight(10)
    def test_initialize_size(self):
        'test_initialize_size: Test that embed.initialize generates a number of embeddings equal to # distinct words'
        embeddings = submitted.initialize(self.data, 5)
        self.assertEqual(len(embeddings), len(set(self.data)), msg="embed.initialize fails to return embedding with a number of entries equal to the number of distinct words in the data")
    
    @weight(10)
    def test_initialize_circle(self):
        'test_initialize_circle: Test that embed.initialize first two dimensions are spaced around a unit circle'
        embeddings = submitted.initialize(self.data, 5)
        self.assertAlmostEqual(np.average([v[0] for v in embeddings.values()]), 0.0, places=2, msg="embed.initialize fails to return embedding in which the first dimension has an average value of 0")
        self.assertAlmostEqual(np.average([v[1] for v in embeddings.values()]), 0.0, places=2, msg="embed.initialize fails to return embedding in which the second dimension has an average value of 0")
        for v in embeddings.values():
            self.assertAlmostEqual(np.linalg.norm(v[:2]), 1.0, places=2, msg="embed.initialize fails to return embedding in which the first two dimensions are on a circle of radius 1")

    @weight(10)
    def test_initialize_nonzero(self):
        'Test that embed.initialize higher dimensions are nonzero'
        embeddings = submitted.initialize(self.data, 5)
        for v in embeddings.values():
            self.assertEqual(len(v), 5, msg="embed.initialize fails to generate embedding vectors of the specified dimension")
            for x in v[2:]:
                self.assertNotEqual(x,0, msg="embed.initialize fails to initialize embeddings whose elements are all nonzero")

    @weight(10)
    def test_gradient_bounds(self):
        'test_gradient_bounds: Test that embed.gradient result is within acceptable bounds'
        embedding = {
            'a': np.ones(4),
            'c': np.zeros(4),
            'b': -np.ones(4)
        }
        data = ['a','a','a','a','a','c','c','c','b','b','b','b','b']
        g = submitted.gradient(embedding, data, 0, 2, 10)
        pa = 1/(1+np.exp(-4))
        pb = 1/(1+np.exp(4))
        
        upper_bound = -2*(1-pa) + 2*pa + 0.01
        lower_bound = -2*(1-pa) - 2*pb - 0.01
        self.assertGreaterEqual(g[0], lower_bound, msg='embed.gradient is producing values that are smaller than theoretically possible')
        self.assertLessEqual(g[0], upper_bound, msg='embed.gradient is producing values that are larger than theoretically possible')

    @weight(10)
    def test_gradient_same_datum(self):
        "test_gradient_same_datum: If there is only one word in the vocabulary, embed.gradient should always produce 2*d*(2*sigma(v*v)-1)*v"
        embedding = {
            'a': np.ones(4)
        }
        data = ['a','a','a','a','a','a','a','a','a']
        g = submitted.gradient(embedding, data, 3, 2, 10)
        pa = 1/(1+np.exp(-np.dot(embedding['a'],embedding['a'])))
        target = 2 * 2 * (2*pa-1) * np.linalg.norm(embedding['a'])
        self.assertAlmostEqual(np.linalg.norm(g), target, places=2, msg='If there is only word in the vocabulary, embed.gradient should always be 2*d*(2*sigma(v*v)-1)*v')
    
    @weight(20)
    def test_gradient_accuracy(self): 
        for i, data in enumerate(self.dataset): 
            np.random.seed(0)
            embeddings = submitted.initialize(data, 10)
            word_pos = range(len(data))
            for j in word_pos: 
                np.random.seed(0)
                g = submitted.gradient(embeddings, data, j, 2, 10)
                ref = self.solution["gradient"][i][j] 
                np.testing.assert_allclose(
                    g, ref, rtol=1e-6, atol=1e-6,
                    err_msg=f"Gradient mismatch: corpus={i}, pos={j}, word={data[j]}"
                )

    @weight(10)
    def test_sgd_nonzero(self):
        'test_sgd_nonzero: Test that embed.sgd updates exactly one vector with a nonzero step'
        embedding = {
            'a': np.array([1,0,0,0,0], dtype='float'),
            'b': np.array([0,1,0,0,0], dtype='float'),
            'c': np.array([0,0,1,0,0], dtype='float'),
            'd': np.array([0,0,0,1,0], dtype='float'),
            'e': np.array([0,0,0,0,1], dtype='float')
        }
        backup = { k:v.copy() for k,v in embedding.items() }
        data = [ 'a','b','c', 'd', 'e' ,'a', 'b', 'c', 'd', 'e']
        embedding = submitted.sgd(embedding, data, 0.01, 1, 2, 10)
        differences = [ np.linalg.norm(embedding[k]-backup[k]) for k in embedding.keys() ]
        self.assertEqual(np.count_nonzero(differences), 1, msg='embed.sgd should produce a nonzero update in the embedding of exactly one word')
    
    @weight(20)
    def test_sgd_accuracy(self): 
        for i, data in enumerate(self.dataset): 
            np.random.seed(0)
            embeddings = submitted.initialize(data, 10)

            # Run SGD
            np.random.seed(0)
            updated = submitted.sgd(embeddings, data, 0.01, 10, 2, 10)
            ref = self.solution["sgd"][i]
            # Compare per key
            self.assertEqual(set(updated.keys()), set(ref.keys()),
                            msg=f"SGD mismatch (keys): corpus={i}")

            for w in updated.keys():
                np.testing.assert_allclose(
                    updated[w], np.asarray(ref[w], dtype=float),
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"SGD mismatch: corpus={i}, word={w}"
                )
