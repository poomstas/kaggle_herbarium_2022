import pandas as pd
import numpy as np
import json

TRAIN_DIR = "/home/jovyan/brian/kaggle_herbarium_2022/data/train_images/"
TEST_DIR = "/home/jovyan/brian/kaggle_herbarium_2022/data/test_images/"

with open("./data/train_metadata.json") as json_file:
    train_meta = json.load(json_file)
with open("./data/test_metadata.json") as json_file:
    test_meta = json.load(json_file)
    
image_ids = [image["image_id"] for image in train_meta["images"]]
image_dirs = [TRAIN_DIR + image["file_name"] for image in train_meta["images"]]
category_ids = [annot["category_id"] for annot in train_meta["annotations"]]
genus_ids = [annot["genus_id"] for annot in train_meta["annotations"] ]

test_ids = [image["image_id"] for image in test_meta]
test_dirs = [TEST_DIR + image["file_name"] for image in test_meta ]

train = pd.DataFrame(data =np.array([image_ids , image_dirs, genus_ids, category_ids ]).T, 
                     columns = ["image_id", "directory","genus_id", "category",])
test = pd.DataFrame(data =np.array([test_ids  , test_dirs ]).T, 
                    columns = ["image_id", "directory",])

train.to_csv("train.csv", index = False)
test.to_csv("test.csv", index = False)