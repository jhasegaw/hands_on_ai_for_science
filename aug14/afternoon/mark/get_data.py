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

def generate_data(n_samples=100, size=20):
    X, Y = [], []
    for _ in range(n_samples):
        img = np.zeros((size, size))
        label = np.random.choice([0, 1]) # 0: bar, 1: blob
        y, x = np.random.randint(5, size-5), np.random.randint(5, size-5)

        if label == 1: # blob
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx**2 + dy**2 <= 4:
                        img[y+dy, x+dx] = 1.0
        else: # bar
            img[y-3:y+4, x] = 1.0

        img += np.random.normal(0, 0.05, img.shape) # Add noise
        X.append(np.clip(img, 0, 1))
        Y.append(float(label))
    return np.array(X), np.array(Y)

def qualitative_check(gt, blob_pos, bar_pos, detection_radius=5):
    """
    Evaluates detection correctness against ground truth.
    Returns: dict with success counts per type.
    """
    results = {'blob': {'total': 0, 'detected': 0}, 'bar': {'total': 0, 'detected': 0}}

    for item in gt:
        gt_type = item['type']
        gt_pos = item['pos']

        # Adjust GT for valid padding offset (center of kernel)
        offset = 2 if gt_type == 'blob' else 3
        adjusted_gt = np.array([gt_pos[0] - offset, gt_pos[1] - offset])

        results[gt_type]['total'] += 1
        preds = blob_pos if gt_type == 'blob' else bar_pos

        if len(preds) > 0:
            distances = np.linalg.norm(preds - adjusted_gt, axis=1)
            if np.any(distances <= detection_radius):
                results[gt_type]['detected'] += 1

    return results
