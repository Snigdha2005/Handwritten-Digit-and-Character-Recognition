import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import preprocessed data
from preprocessing import flattened_images, valid_labels

# Check if there are any successfully preprocessed images
if len(flattened_images) == 0:
    print("No images were successfully preprocessed. Check the image paths and preprocessing steps.")
    exit()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(valid_labels)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(flattened_images)

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# Plot the data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=encoded_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Classes')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP of Handwritten Digits and Characters')
plt.show()
