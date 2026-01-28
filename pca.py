import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load Data (4 dimensions: length and width of petals/sepals)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Standardize (Crucial! PCA needs data to be centered around zero)
# This is like making sure all your sensors are on the same scale.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Apply PCA
# We want to reduce 4 columns down to 2 "Principal Components"
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# 4. Create a new DataFrame with the "compressed" data
pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])

print(f"Original shape: {scaled_data.shape}")
print(f"Reduced shape: {pca_df.shape}")
print(f"Explained Variance: {pca.explained_variance_ratio_}")