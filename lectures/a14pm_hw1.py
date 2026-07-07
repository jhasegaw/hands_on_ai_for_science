import numpy as np

def convolve2d(image, kernel):
    """
    Computes the 2D convolution of an image with a given kernel.

    Args:
        image (np.ndarray): 2D array of shape (H, W)
        kernel (np.ndarray): 2D array of shape (Kh, Kw)

    Returns:
        np.ndarray: The convolved feature map without any padding.

    IMPORTANT:
      - No padding (!!! Spacial size of output feature map is different from image)
      - Stride = 1

    """

    H, W = image.shape
    kH, kW = kernel.shape

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Input image  and kernel must be 2D.")
    if kH > H or kW > W:
        raise ValueError("Kernel cannot be larger than the input in any dimension.")

    response = None

    # TODO:  Your implementation here

    raise NotImplementedError

    return response


def correlation2d(image, kernel):
    """
    Computes the 2D cross-correlation of an image with a given kernel.

    Args:
        image (np.ndarray): 2D array of shape (H, W)
        kernel (np.ndarray): 2D array of shape (Kh, Kw)

    Returns:
        np.ndarray: The cross-correlation feature map without any padding.

    IMPORTANT:
      - No padding (!!! Spacial size of output is different from image)
      - No stride other than 1

    """

    H, W = image.shape
    kH, kW = kernel.shape

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Input image and kernel must be 2D.")
    if kH > H or kW > W:
        raise ValueError("Kernel cannot be larger than the input in any dimension.")

    response = None

    # TODO:  Your implementation here

    raise NotImplementedError

    return response

# Part 2

def detect_blob(image, k=5):
    """
    Returns a simple kxk kernel designed to detect blobs (normalize).
    The kernel size and shape depend on k.

    You can use as blob radius = (center)**2
    """
    kernel = None
    positions = None
    # TODO:  Your implementation here

    raise NotImplementedError

    return kernel, positions

def detect_bar(image, k=7):
    """
    Returns a kxk kernel designed to detect vertical bars.
    All values in filter should be 0 except central vertical line
    """
    kernel = None
    positions = None

    # TODO:  Your implementation here
    raise NotImplementedError

    return kernel, positions

def detect_positions(feature_map):
    """
    Identifies coordinates where the response is the highest
    """
    # TODO:  Your implementation here
    positions = (0,0)
    raise NotImplementedError

    return positions

# Part 3

def detect_bar_gradient(image, scale=2):
    """
    Returns a 3x3 Horizontal Sobel kernel and the coordinates of global maxima
    after convolving with the input image.
    """
    # TODO:  Your implementation here
    raise NotImplementedError

    # Define a standard 3x3 Horizontal Sobel filter
    kernel = None
    positions = None

    # Perform convolution
    # Detect positions of global maxima

    return kernel, positions

# Part 4

def train_classifier(images, labels, epochs=100, lr=0.01):
    """
    Optimizes a 3x3 kernel to distinguish between two patterns (e.g., blobs vs bars).

    Args:
        images (np.ndarray): Array of shape (N, H, W) containing training images.
        labels (np.ndarray): Array of shape (N,) containing 1.0 (target) or 0.0 (other).
        epochs (int): Number of times to iterate over the dataset.
        lr (float): Learning rate for gradient descent.

    Returns:
        np.ndarray: The learned 3x3 kernel.
    """

    # TODO:  Your implementation here
    raise NotImplementedError

    # 1. Initialization
    # Initialize a 3x3 kernel with small random values (e.g., mean 0, std 0.01)
    kernel = None


    # --- 2. Forward Pass ---
    # Perform convolution using your convolve2d implementation


    # Global Max Pooling: Find the maximum value (prediction)
    # and its (row, col) coordinates in the feature map
    # Hint: np.argmax and np.unravel_index maybe useful

    # --- 3. Compute Loss ---
    # Calculate Mean Squared Error

    # --- 4. Backpropagation (Simplified) ---
    # Calculate the gradient of the loss with respect to the kernel.
    # Use the chain rule

    # --- 5. Update Weights ---
    # Apply the gradient descent update rule: K = K - (lr * gradient)


    return kernel

def predict(image, kernel, threshold=0.5):
    """
    Uses a trained kernel to recognize if a pattern is present.
    """
    # TODO:  Your implementation here
    raise NotImplementedError

    max_response = None
    # 1. Convolve image with learned kernel
    # 2. Find max response
    return max_response > threshold, max_response
