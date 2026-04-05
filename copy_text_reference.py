import os

BASE_PATH = "data/train"
TARGET_IMG_PATH = "target"

DATASET_SOURCE_PATH = os.path.join("..", "Capstone", "dataset", "polygon_224", "train")
DATASET_REF_PATH = os.path.join(DATASET_SOURCE_PATH, "reference")
DATASET_TEXT_PATH = os.path.join(DATASET_SOURCE_PATH, "text")

target_list = os.listdir(os.path.join(BASE_PATH, TARGET_IMG_PATH))

for target in target_list:
    
    target_name = target.split(".")[0]
    print(f"File Name: {target_name}")
    
    # Copy reference image
    source_ref_path = os.path.join(DATASET_REF_PATH, f"{target_name}.png")
    target_ref_path = os.path.join(BASE_PATH, "reference", f"{target_name}.png")
    # os.makedirs(os.path.dirname(target_ref_path), exist_ok=True)
    os.system(f"cp {source_ref_path} {target_ref_path}")
    
    # Copy text
    source_txt_path = os.path.join(DATASET_TEXT_PATH, f"{target_name}.txt")
    target_txt_path = os.path.join(BASE_PATH, "text", f"{target_name}.txt")
    # os.makedirs(os.path.dirname(target_txt_path), exist_ok=True)
    os.system(f"cp {source_txt_path} {target_txt_path}")
