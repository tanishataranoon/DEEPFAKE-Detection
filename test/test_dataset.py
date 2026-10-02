"""
===========================================
Test DeepfakeDataset
===========================================
"""

from torch.utils.data import DataLoader

from data.dataset.appearance.deepfake_dataset import DeepfakeDataset


def main():

    dataset = DeepfakeDataset(
        csv_file="data/split/train.csv",
        video_root="dataset",
    )

    print("=" * 60)
    print("Dataset Test")
    print("=" * 60)

    print("Dataset size :", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    images, labels = next(iter(loader))

    print("Batch image shape :", images.shape)
    print("Batch labels      :", labels)

    print("Min pixel :", images.min().item())
    print("Max pixel :", images.max().item())

    print("\nDataset test passed.")


if __name__ == "__main__":
    main()