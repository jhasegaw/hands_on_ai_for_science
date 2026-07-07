import os 
import numpy as np

def load_mnist_numpy(path="data/"):
    """
    Load MNIST dataset from previously saved numpy arrays.
    
    Returns:
        x_train: np.array of shape (60000, 784), normalized to [0,1]
        y_train: np.array of shape (60000,), label encoded
        x_test: np.array of shape (10000, 784), normalized to [0,1]
        y_test: np.array of shape (10000,), label encoded
    """
    # Check that the files exist
    required_files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    for f in required_files:
        if not os.path.exists(os.path.join(path, f)):
            raise FileNotFoundError(f"{f} not found in {path}. Make sure you saved MNIST first.")

    # Load arrays (normalize between 0 and 1)
    x_train = np.load(os.path.join(path, "x_train.npy")) / 255.
    y_train = np.load(os.path.join(path, "y_train.npy")).astype(int)
    x_test = np.load(os.path.join(path, "x_test.npy"))/ 255. 
    y_test = np.load(os.path.join(path, "y_test.npy")).astype(int)

    return x_train, y_train, x_test, y_test