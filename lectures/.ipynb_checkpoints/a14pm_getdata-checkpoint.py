import numpy as np

def generate_synthetic_data(width=30, height=30, num_shapes=2):
    """
    Generates a synthetic grayscale image with random blobs and bars.
    Returns:
        image: The noisy synthetic image.
        ground_truth: A list of dicts with shape type and center coordinates.
    """
    image = np.zeros((height, width))
    ground_truth = []
    shape_types = ['blob', 'bar']

    for i in range(num_shapes):
        # Randomly choose shape type: 0 for blob, 1 for bar
        shape_type = shape_types[i]
        y, x = np.random.randint(10, height-10), np.random.randint(10, width-10)

        if shape_type == 'blob':
            # Create a 5x5 circular blob
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx**2 + dy**2 <= 4:
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            image[y+dy, x+dx] = 1.0
            ground_truth.append({'type': 'blob', 'pos': (y, x)})

        else:
            # Create a 1x7 vertical bar
            for dy in range(-3, 4):
                if 0 <= y+dy < height:
                    image[y+dy, x] = 1.0
            ground_truth.append({'type': 'bar', 'pos': (y, x)})

    # Add Gaussian noise
    image += np.random.normal(0, 0.1, image.shape)
    image = np.clip(image, 0, 1)

    return image, ground_truth
