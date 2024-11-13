from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parameters for synthetic data
n_samples = [50000, 30000, 20000, 10000]  # Different cluster sizes
cluster_std = [1.0, 2.5, 0.5, 3.0]        # Varying densities
n_features = 20                            # High dimensionality

# Random centers in high-dimensional space
centers = np.random.uniform(-50, 50, size=(len(n_samples), n_features))

# Generate synthetic dataset
X, y = make_blobs(n_samples=n_samples, centers=centers,
                  cluster_std=cluster_std, n_features=n_features,
                  random_state=42)

# Add uniformly distributed noise/outliers
n_noise = 10000  # Number of noise points
noise = np.random.uniform(-100, 100, size=(n_noise, n_features))

# Combine the datasets
X = np.vstack((X, noise))
y = np.hstack((y, [-1]*n_noise))  # Label noise points as -1


# Create a noisy DataFrame
df_noisy = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df_noisy['cluster'] = y

# Save noisy dataset to CSV
df_noisy.to_csv('../Datasets/synthetic_dataset.csv', index=False)