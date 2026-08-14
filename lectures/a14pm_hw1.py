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

    response = np.zeros((H-kH+1, W-kW+1))

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

    response = np.zeros((H-kH+1, W-kW+1))

    # TODO:  Your implementation here

    raise NotImplementedError

    return response

# Part 2

def detect_blob(image, k=5):
    """
    Returns a simple kxk kernel designed to detect blobs (normalize).
    The kernel size and shape depend on k.

    You can use as blob radius**2 = (center)**2
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

    #raise NotImplementedError
    positions = np.argmax(feature_map)
    positions = np.unravel_index(positions, feature_map.shape)
    positions = np.array([list(positions)])
    # positions2 = np.argwhere(feature_map == np.max(feature_map))
    # print("-----", positions, "-----", positions2)

    return positions

# Part 3

def detect_bar_gradient(image, scale=2):
    """
    Returns a 3x3 Horizontal Derivative Sobel kernel and the coordinates of global maxima
    after convolving with the input image.
    """
    kernel = None
    positions = None

    # TODO:  Your implementation here

    raise NotImplementedError

    # Define a standard 3x3 Horizontal Derivative Sobel filter
    kernel = np.array([[-1, 0, 1],[-scale, 0, scale], [-1, 0, 1]])
    # Perform convolution
    response = convolve2d(image, kernel)
    # Detect positions of global maxima
    positions = detect_positions(response)

    return kernel, positions

# Part 4

def train_classifier(images, labels, epochs=100, lr=0.01):
    """
    Optimizes a 3x3 kernel to distinguish between two patterns (e.g., blobs vs bars).

    Args:
        images (np.ndarray): Array of shape (N, H, W) containing training images.
        labels (np.ndarray): Array of shape (N,) containing 1.0 (blobs) or 0.0 (bars).
        epochs (int): Number of times to iterate over the dataset.
        lr (float): Learning rate for gradient descent.

    Returns:
        np.ndarray: The learned 3x3 kernel.
    """

    # TODO:  Your implementation here

    raise NotImplementedError

    # 1. Initialization
    # Initialize a 3x3 kernel with small random values
    kernel = None
    # kernel = np.random.rand(3,3)
    kernel = np.random.normal(0, 0.01, (3, 3))
    # return kernel
    # exit()


    for epoch in range(0, epochs):
        for i in range(0, len(images)):
    # --- 2. Forward Pass ---
    # Perform convolution using your convolve2d implementation
            img, y =  images[i], labels[i]
            feature_map = convolve2d(img, kernel)
    # Global Max Pooling: Find the maximum value (prediction)
    # and its (row, col) coordinates in the feature map
            positions = detect_positions(feature_map)

            acc_grad = np.zeros((3,3))

            for position in positions:
                # print(position)
                predicted = feature_map[position[0], position[1]]
                # print(predicted)

    # --- 3. Compute Loss ---
    # Calculate Mean Squared Error

                error = (predicted-y)**2

    # --- 4. Backpropagation (Simplified) ---
    # Calculate the gradient of the loss with respect to the kernel.
    # Use the chain rule
                gradient_loss = 2*(predicted-y)
                image_patch = img[position[0]:position[0]+3, position[1]:position[1]+3]
                gradient_max_conv = gradient_loss*image_patch
                acc_grad+=gradient_max_conv

    # --- 5. Update Weights ---
    # Apply the gradient descent update rule: K = K - (lr * gradient)
            acc_grad = np.rot90(acc_grad, 2)
            # print("------", len(positions))
            kernel -=(lr * acc_grad/(len(positions)))


    return kernel

def predict(image, kernel, threshold=0.5):
    """
    Uses a trained kernel to recognize if a pattern is present.
    """
    # TODO:  Your implementation here

    raise NotImplementedError

    max_response = None
    # 1. Convolve image with learned kernel
    map = convolve2d(image, kernel)
    # 2. Find max response
    max_response = np.max(map)
    return max_response > threshold, max_response
