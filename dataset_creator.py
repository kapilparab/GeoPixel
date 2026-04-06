import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

import os
import json

def mask_to_rle(mask_path):
    # 1. Load the binary mask image
    mask_img = Image.open(mask_path).convert('L')
    mask_array = np.array(mask_img)

    # 2. Convert to strict 0s and 1s
    binary_mask = (mask_array > 128).astype(np.uint8)
    
    # 3. pycocotools requires Fortran-contiguous arrays
    binary_mask = np.asfortranarray(binary_mask)

    # 4. Encode to RLE
    rle = mask_utils.encode(binary_mask)
    
    # 5. Decode the bytes object to a string for JSON saving
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle

DATASET_DEST_PATH = "data/train"
REFERENCE_DIR =  "reference"
TARGET_DIR = "target"
TEXT_DIR = "text"

DATASET_SOURCE_PATH = "../Capstone/dataset/polygon_224/train"

GOOGLE_DRIVE_BASE_PATH = "/content/drive/MyDrive/Capstone/dataset"

result = []

img_list = os.listdir(os.path.join(DATASET_SOURCE_PATH, "reference"))
print(f"Total images found: {len(img_list)}")

for i, img in enumerate(img_list):
    
    if i == 8000:
        break

    ann = {}

    img_path = os.path.join(DATASET_SOURCE_PATH, REFERENCE_DIR, img)
    img_name = img.split(".")[0]
    
    target_path = os.path.join(DATASET_SOURCE_PATH, TARGET_DIR, img_name + ".png")
    
    if not os.path.isfile(target_path):
        print(f"Target mask not found for {img_name}, skipping.")
        break
    
    text_path = os.path.join(DATASET_SOURCE_PATH, TEXT_DIR, img_name + ".txt")
    
    if not os.path.isfile(text_path):
        print(f"Text prompt not found for {img_name}, skipping.")
        break
    
    dest_img_path = os.path.join(GOOGLE_DRIVE_BASE_PATH, REFERENCE_DIR, img)
    
    # Read the text prompt
    with open(text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
        
    full_instruction = f"Can you segment the TARGET based on this description: {prompt_text}"
    
    target_rle = mask_to_rle(target_path)
    
    ann["id"] = i + 1
    ann["conversations"] = [
        {
            "from": "human",
            "value": full_instruction 
        },
        {
            "from": "bot",
            "value": "Sure, here is the segmentation mask for the requested area. <p> TARGET </p> [SEG] ."
        }
    ]

    ann["image"] = dest_img_path
    
    ann["segmentation"] = [
        {
            "size": target_rle["size"],
            "counts": target_rle["counts"]
        }
    ]
    result.append(ann)

output_json_path = "data/train/dataset.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print(f"Successfully processed {len(result)} images and saved to {output_json_path}")