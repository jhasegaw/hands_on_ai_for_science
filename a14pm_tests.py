import unittest
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight, visibility
from submitted import convolve2d, correlation2d, detect_blob, detect_bar, detect_bar_gradient, train_classifier, predict
from get_data import generate_data

class TestSpatialOperations(unittest.TestCase):

    def setUp(self):
        self.image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        self.kernel = np.array([
            [1, 0],
            [0, -1]
        ], dtype=float)

    @weight(5)
    def test_correlation_values(self):
        """Tests cross-correlation with a known toy example."""
        # Expected calculation for correlation (no flip):
        # (1*1 + 2*0 + 4*0 + 5*-1) = -4
        # (2*1 + 3*0 + 5*0 + 6*-1) = -4
        # (4*1 + 5*0 + 7*0 + 8*-1) = -4
        # (5*1 + 6*0 + 8*0 + 9*-1) = -4
        expected = np.array([
            [-4.0, -4.0],
            [-4.0, -4.0]
        ])
        result = correlation2d(self.image, self.kernel)

        self.assertIsNotNone(result, "correlation2d returned None")
        np.testing.assert_allclose(result, expected, err_msg="Correlation values are incorrect.")

    @weight(5)
    def test_convolution_values(self):
        """Tests convolution (with kernel flip) with a known toy example."""
        # Expected calculation for convolution (flipped kernel [[-1, 0], [0, 1]]):
        # (1*-1 + 2*0 + 4*0 + 5*1) = 4
        expected = np.array([
            [4.0, 4.0],
            [4.0, 4.0]
        ])
        result = convolve2d(self.image, self.kernel)

        self.assertIsNotNone(result, "convolve2d returned None")
        np.testing.assert_allclose(result, expected, err_msg="Convolution values are incorrect. Did you flip the kernel?")

    @weight(5)
    def test_output_shapes(self):
        """Verifies valid padding shapes for both operations."""
        img = np.zeros((10, 10))
        ker = np.zeros((3, 3))
        # 10 - 3 + 1 = 8

        self.assertEqual(correlation2d(img, ker).shape, (8, 8))
        self.assertEqual(convolve2d(img, ker).shape, (8, 8))

    @weight(5)
    def test_relationship(self):
        """Tests if convolve2d(I, K) is equal to correlation2d(I, flipped_K)."""
        img = np.random.rand(5, 5)
        ker = np.random.rand(3, 3)

        conv_res = convolve2d(img, ker)
        # Manually flip kernel for correlation
        flipped_ker = ker[::-1, ::-1]
        corr_res = correlation2d(img, flipped_ker)

        np.testing.assert_allclose(conv_res, corr_res, err_msg="Convolution must equal Correlation with a flipped kernel.")

    @weight(5)
    def test_identity_kernel(self):
        """
        Tests that an identity kernel (flipped 1 at center) returns
        the original image sub-region.
        """

        image = np.random.rand(5, 5)
        # A 3x3 kernel that is 0 everywhere except the very center (which stays center when flipped)
        kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=float)

        # For a 5x5 image and 3x3 kernel, valid output is 3x3.
        # It should match the center 3x3 of the original image.
        expected = image[1:4, 1:4]

        result = convolve2d(image, kernel)
        np.testing.assert_allclose(result, expected, err_msg="Identity kernel failed to return image sub-region.")

    @weight(5)
    def test_blob_kernel_generation(self):
        """Checks if detect_blob generates a kernel of the requested size."""
        k_size = 9
        image = np.zeros((20, 20))
        kernel, _ = detect_blob(image, k=k_size)

        self.assertEqual(kernel.shape, (k_size, k_size))
        self.assertAlmostEqual(np.sum(kernel), 1.0, msg="Kernel must be normalized.")
        # Check center is 1
        self.assertEqual(kernel[k_size//2, k_size//2], 1.0 / np.sum(kernel > 0))

    @weight(5)
    def test_detection_at_different_scales(self):
        """Tests if a larger kernel can find a larger blob."""
        size = 100
        image = np.zeros((size, size))

        # Place a large 11x11 blob
        by, bx = 50, 50
        y, x = np.ogrid[-10:11, -10:11]
        mask = x**2 + y**2 <= 5**2
        image[by-10:by+11, bx-10:bx+11][mask] = 1.0

        # Use an 11x11 kernel
        _, pos = detect_blob(image, k=11)

        # Expected position in valid space: 50 - 11//2 = 45
        expected_pos = np.array([45, 45])
        distances = np.linalg.norm(pos - expected_pos, axis=1)
        self.assertTrue(np.any(distances < 2), f"Failed to detect large blob. Found: {pos}")

    @weight(5)
    def test_bar_kernel_size(self):
        """Checks if bar kernel generates correct vertical line."""
        k_size = 3
        image = np.zeros((10, 10))
        kernel, _ = detect_bar(image, k=k_size)

        expected_kernel = np.array([
            [0, 1/3, 0],
            [0, 1/3, 0],
            [0, 1/3, 0]
        ])
        np.testing.assert_allclose(kernel, expected_kernel)

    @weight(5)
    def test_kernel_construction_sobel(self):
        """Verifies that scale=2 produces a standard Sobel kernel."""
        # Create a dummy image as the function requires one
        image = np.zeros((10, 10))
        kernel, _ = detect_bar_gradient(image, scale=2)
        expected_sobel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=float)

        np.testing.assert_allclose(kernel, expected_sobel, err_msg="Sobel kernel (scale=2) is incorrect.")

    @weight(5)
    def test_kernel_construction_prewitt(self):
        """Verifies that scale=1 produces a standard Prewitt kernel."""
        # Create a dummy image as the function requires one
        image = np.zeros((10, 10))
        kernel, _ = detect_bar_gradient(image, scale=1)
        expected_prewitt = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=float)

        np.testing.assert_allclose(kernel, expected_prewitt, err_msg="Prewitt kernel (scale=1) is incorrect.")

    @weight(10)
    def test_edge_localization(self):
        """Tests if the gradient filter correctly localizes a vertical edge."""
        # Generate synthetic image with a vertical edge at column 5
        image = np.zeros((10, 10))
        image[:, 5] = 1.0

        # Expected position in 'valid' output: 5 - (3 // 2) = 4
        expected_col = 5

        # Test with Sobel scale
        _, pos = detect_bar_gradient(image, scale=2)

        self.assertTrue(len(pos) > 0, "No maxima detected by gradient filter.")

        # Every row in the output at the transition column should be a maximum
        for p in pos:
            self.assertEqual(p[1], expected_col,
                             f"Edge detected at column {p[1]}, expected {expected_col}")

    @weight(5)
    def test_noise_robustness(self):
        """Verifies that higher scale (Sobel) maintains the same peak location with noise."""
        # Generate synthetic image with a vertical edge at column 5
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0
        expected_col = 4

        noisy_image = image + np.random.normal(0, 0.05, image.shape)
        _, pos = detect_bar_gradient(noisy_image, scale=2)

        # The peak should still be centered around the expected column
        mean_col = np.mean(pos[:, 1])
        self.assertAlmostEqual(mean_col, expected_col, delta=3)

    @weight(5)
    def test_learning_convergence(self):
        """
        Check if the training loop actually updates weights.
        We initialize a kernel, train it briefly, and ensure it's different.
        """
        # Set seed for reproducibility in this specific test
        np.random.seed(42)
        X, y = generate_data(10, size=15)

        # We'll manually check if weights change from a fixed initial state
        # (Though train_classifier initializes its own, we test that the result isn't just zeros/random)
        kernel = train_classifier(X, y, epochs=5, lr=0.1)

        # A trained kernel should not be all zeros or perfectly identical to
        # a standard normal distribution anymore.
        self.assertFalse(np.allclose(kernel, 0), "Kernel weights did not update.")

    @weight(5)
    def test_predict_thresholding(self):
        """Test if the threshold correctly determines boolean output."""
        img = np.zeros((10, 10))
        img[4:7, 4:7] = 1.0 # Strong center pattern
        kernel = np.ones((3, 3)) # Will produce a high response

        # Score will be 9.0 (sum of ones in 3x3 patch)
        is_pattern_high, _ = predict(img, kernel, threshold=5.0)
        is_pattern_low, _ = predict(img, kernel, threshold=10.0)

        self.assertTrue(is_pattern_high)
        self.assertFalse(is_pattern_low)

    @weight(15)
    def test_model_accuracy(self):
        """
        End-to-end test: Verify if the model can achieve > 85% accuracy.
        This ensures the backpropagation and update logic are correct.
        """
        np.random.seed(42)
        # Generate enough data for a meaningful test
        X, y = generate_data(n_samples=200, size=20)

        # Split
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        # Train
        kernel = train_classifier(X_train, y_train, epochs=50, lr=0.0001)

        # Evaluate
        correct = 0
        for i in range(len(X_test)):
            is_blob, _ = predict(X_test[i], kernel)
            if is_blob == bool(y_test[i]):
                correct += 1

        accuracy = correct / len(X_test)
        self.assertGreater(accuracy, 0.85, f"Accuracy too low: {accuracy:.2f}. Check gradient/update logic.")
