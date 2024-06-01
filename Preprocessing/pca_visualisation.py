import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import preprocessed data
from preprocessing import flattened_images, labels

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels[:len(flattened_images)])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(flattened_images)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=encoded_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Classes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Handwritten Digits and Characters')
plt.show()
