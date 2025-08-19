import kagglehub

# Download latest version
# path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled")
path = kagglehub.dataset_download("mitkir/shapenet")

print("Path to dataset files:", path)