import os
import re
import cv2
import torch
import random
import numpy as np
import transformers 
from peft import PeftModel
from model.geopixel import GeoPixelForCausalLM

# ==========================================
# 1. Configuration (Update these paths!)
# ==========================================
BASE_MODEL = "MBZUAI/GeoPixel-7B-RES" # Set to "MBZUAI/GeoPixel-7B" if you didn't do the RES task
LORA_PATH = "/content/drive/MyDrive/Capstone/GeoPixel/checkpoint-last"
TEST_IMAGE_PATH = "/content/GL256_333.png"
TEST_PROMPT = """
Can you segment the TARGET based on this description: TARGET is one of the districts of GREEN, in the historical region of BLUE. Its capital and largest city is YELLOW. Jafara borders MAGENTA in northeast, CYAN in south and ORANGE in the west.
"""

VIS_SAVE_PATH = "./vis_output"

os.makedirs(VIS_SAVE_PATH, exist_ok=True)

# ==========================================
# 2. Tokenizer Setup
# ==========================================
print(f'Initializing tokenizer from: {BASE_MODEL}')
tokenizer = transformers.AutoTokenizer.from_pretrained(
    BASE_MODEL,
    padding_side='right',
    use_fast=False,
    trust_remote_code=True,
)

# Adding all special tokens required by GeoPixel's chat.py
special_tokens = ['[SEG]', '<p>', '</p>']
tokenizer.add_tokens(special_tokens, special_tokens=True)
tokenizer.pad_token = tokenizer.unk_token

seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

# ==========================================
# 3. Base Model Initialization
# ==========================================
print("Loading Base GeoPixel Model...")
geo_model_args = {
    "vision_pretrained": 'facebook/sam2-hiera-large',
    "seg_token_idx" : seg_token_idx, 
    "bop_token_idx" : bop_token_idx, 
    "eop_token_idx" : eop_token_idx  
}

base_model = GeoPixelForCausalLM.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    **geo_model_args
).cuda()

base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id

base_model.resize_token_embeddings(len(tokenizer))
base_model.tokenizer = tokenizer

# ==========================================
# 4. Inject LoRA Adapter
# ==========================================
print("Injecting Custom LoRA Weights...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("Model Ready!")

# ==========================================
# 5. Run Inference & Save Mask
# ==========================================
print(f"Processing image: {TEST_IMAGE_PATH}")

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Notice we use `evaluate` and pass the image path in a list, not a PIL Image
    response, pred_masks = model.evaluate(tokenizer, TEST_PROMPT, images=[TEST_IMAGE_PATH], max_new_tokens=300)

if pred_masks and '[SEG]' in response:
    print("\nMasks generated! Rendering image...")
    pred_masks = pred_masks[0].detach().cpu().numpy() > 0
    
    # Load original image via OpenCV
    image_np = cv2.imread(TEST_IMAGE_PATH)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    save_img = image_np.copy()
    
    pattern = r'<p>(.*?)</p>\s*\[SEG\]'
    matched_text = re.findall(pattern, response)
    
    # Apply color overlays for each predicted mask
    for i in range(pred_masks.shape[0]):
        mask = pred_masks[i]
        color = [random.randint(0, 255) for _ in range(3)]
        mask_rgb = np.stack([mask, mask, mask], axis=-1) 
        color_mask = np.array(color, dtype=np.uint8) * mask_rgb

        # Blend the mask with the original image
        save_img = np.where(mask_rgb, 
                (save_img * 0.5 + color_mask * 0.5).astype(np.uint8), 
                save_img)

    # Save the output
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    filename = TEST_IMAGE_PATH.split("/")[-1].split(".")[0]
    save_path = f"{VIS_SAVE_PATH}/{filename}_masked.jpg"
    cv2.imwrite(save_path, save_img)
    print(f"\nSUCCESS: Masked image saved to {save_path}")
    
    # Print the model's text response cleanly
    clean_response = re.sub(r'<p>(.*?)</p>', '', response).replace('[SEG]', '').replace("  ", " ").strip()
    print(f"Model Description: {clean_response}")
else:
    print("\nNo masks were found in the output. Here is the raw text response:")
    print(response.replace("\n", "").replace("  ", " "))