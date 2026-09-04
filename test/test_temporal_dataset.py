"""
=========================================================
Test Temporal Dataset
=========================================================
Quick sanity check: does the dataset load one sample
without crashing, and is the shape correct?
=========================================================
"""

from data.raw.temporal.temporal_dataset import TemporalDeepfakeDataset

CSV = "data/split/train.csv"
ROOT = "data/raw"


def main():
    dataset = TemporalDeepfakeDataset(
        metadata_csv=CSV,
        dataset_root=ROOT,
        image_size=224,
        num_frames=16,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Clip shape :", sample["clip"].shape)   # expect (16, 3, 224, 224)
    print("Label      :", sample["label"].item())
    print("Video name :", sample["video_name"])

    print("\nMin pixel:", sample["clip"].min().item())
    print("Max pixel:", sample["clip"].max().item())
    
    import torchvision.transforms as T
    frame0 = sample["clip"][0]  # first frame of the clip, shape (3, 224, 224)
    T.ToPILImage()(frame0).save("test/sample_crop.png")
    print("\nSaved test/sample_crop.png — open it and check it's a face")


if __name__ == "__main__":
    main()