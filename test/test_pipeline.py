import pandas as pd

train = pd.read_csv("data/split/train.csv")
val = pd.read_csv("data/split/val.csv")
test = pd.read_csv("data/split/test.csv")
meta = pd.read_csv("data/processed/metadata.csv")

print(len(train))
print(len(val))
print(len(test))
print(len(meta))

print(len(train)+len(val)+len(test))


cols = [
    "video_name",
    "track_id",
    "clip_id"
]

all_df = pd.concat([train, val, test], ignore_index=True)

duplicates = all_df.duplicated(cols)

print("Duplicate clips:", duplicates.sum())
import os
import re

def get_source_group(video_name):

    name = os.path.splitext(video_name)[0]

    if re.match(r"^\d+__", name) and name.count("__") == 1:
        return name

    match = re.match(r"^\d+_(\d+)__(.+?)__[^_]+$", name)

    if match:
        source_id = match.group(1)
        scenario = match.group(2)
        return f"{source_id}__{scenario}"

    return name
train_groups = set(train["video_name"].apply(get_source_group))
val_groups = set(val["video_name"].apply(get_source_group))
test_groups = set(test["video_name"].apply(get_source_group))

print(len(train_groups & val_groups))
print(len(train_groups & test_groups))
print(len(val_groups & test_groups))