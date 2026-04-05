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

DATASET_BASE_PATH = "data/train"
REFERENCE_DIR = os.path.join(DATASET_BASE_PATH, "reference")
TARGET_DIR = os.path.join(DATASET_BASE_PATH, "target")
TEXT_DIR = os.path.join(DATASET_BASE_PATH, "text")

images = os.listdir(REFERENCE_DIR)
result = []

for i, img in enumerate(images):
    
    if not img.endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    ann = {}
    
    img_path = os.path.join(REFERENCE_DIR, img)
    img_name = img.split(".")[0]
    
    target_path = os.path.join(TARGET_DIR, img_name + ".png")
    text_path = os.path.join(TEXT_DIR, img_name + ".txt")
    
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
    
    ann["image"] = img_path
    
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