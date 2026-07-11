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
    response = correlation2d(image, kernel[::-1,::-1])
    # TODO:  Your implementation here

    #raise NotImplementedError

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
    response = np.zeros((H-kH+1,W-kW+1))
    for row in range(H-kH+1):
        for col in range(W-kW+1):
            response[row,col] = np.sum(kernel*image[row:row+kH,col:col+kW])
    #raise NotImplementedError

    return response

# Part 2

def detect_positions(image, kernel):
    """
    Convolve an image with a kernel, then return the coordinates at which the 
    convolution has its maximum.

    @param:
    image - an image
    kernel - a kernel

    @return:
    positions - a tuple of (row, column) indices of the location with highest response
    """
    featuremap = convolve2d(image, kernel)
    return np.unravel_index(np.argmax(featuremap), featuremap.shape)
    
def detect_blob(image, k=5):
    """
    Returns a simple kxk kernel designed to detect blobs (normalize).
    The kernel size and shape depend on k.

    You can use as blob radius = (k-1)//2,
    blob center = [(k-1)//2,(k-1)//2]

    @param:
    image - a 2d grayscale image
    k - the size of the kernel

    @return:
    kernel - All values in kernel should be 0 except central vertical line
    positions - a tuple with the best-matching (row,column) coordinate
    """
    kernel = None
    positions = None
    # TODO:  Your implementation here
    kernel = np.zeros((k,k))
    radius = (k-1)//2
    center = (k-1)//2
    for row in range(k):
        for col in range(k):
            if (row-center)**2 + (col-center)**2 <= radius**2:
                kernel[row,col] = 1
    kernel /= np.sum(kernel)
    positions = detect_positions(image, kernel)
    #raise NotImplementedError

    return kernel, positions

def detect_bar(image, k=7):
    """
    Returns a kxk kernel designed to detect vertical bars,
    and the (row,column) coordinate in the image that best matches.

    @param:
    image - a 2d grayscale image
    k - the size of the kernel

    @return:
    kernel - All values in kernel should be 0 except central vertical line
    positions - a tuple with the best-matching (row,column) coordinate
    """
    kernel = None
    positions = None

    # TODO:  Your implementation here
    #raise NotImplementedError
    kernel = np.zeros((k,k))
    center = (k-1)//2
    kernel[:,center] = 1/k
    positions = detect_positions(image, kernel)
    return kernel, positions

