# %%
# Load libraries and data
import pandas as pd
from pathlib import Path

ROOT = Path("CompSci Repo\ITEC5920\ITEC5920\Project\CrisisMMD")
IMAGES = ROOT / "CrisisMMD_v2.0"
SPLITS = ROOT / "crisismmd_datasplit_all"

train_tsv = SPLITS / "task_informative_text_img_train.tsv"
dev_tsv = SPLITS / "task_informative_text_img_dev.tsv"
test_tsv = SPLITS / "task_informative_text_img_test.tsv"

def load_split(data_path):

    features = pd.read_csv(data_path, sep="\t")

    # label_image for images which is then converted to binary
    features["informative"] = features["label_image"].map({"informative": 1, "not_informative": 0})

    features["img_path"] = features["image"].apply(lambda p: str((IMAGES / p).resolve()))

    features = features.dropna(subset=["informative", "img_path"])
    return features[["tweet_id", "image_id", "img_path", "informative"]]

train = load_split(train_tsv)
dev = load_split(dev_tsv)
test = load_split(test_tsv)

#Sanity Checks for Data Loading
print(train.head())
print(train["informative"].value_counts())

missing = train[~train["img_path"].apply(lambda p: Path(p).exists())]
print("Missing images:", len(missing))
print(missing.head(3))

# %%
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

# %%
import sys
print(sys.executable)
# %%
