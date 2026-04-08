import os
import json

NEW_PATH = "/home/u21/kapilparab/geopixel_train"

with open('data/train/dataset.json', 'r') as f:
    data = json.load(f)
    
for item in data:
    
    img_path = item["image"]
    
    fname = img_path.rpartition('/')[2]
    
    new_img_path = os.path.join(NEW_PATH, fname)

    item['image'] = new_img_path

with open('data/train/dataset_hpc.json', 'w') as f:
    json.dump(data, f, indent=4)