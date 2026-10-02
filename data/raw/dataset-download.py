import kagglehub
import shutil
from pathlib import Path

# Download dataset using KaggleHub
download_path = kagglehub.dataset_download(
    "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"
)


destination = Path("data/raw/DFD")

destination.mkdir(parents=True, exist_ok=True)


shutil.copytree(download_path, destination, dirs_exist_ok=True)

print("Dataset downloaded and copied to:")
print(destination.resolve())