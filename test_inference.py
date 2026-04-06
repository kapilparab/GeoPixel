import torch
import transformers
from peft import PeftModel
from PIL import Image

# Import the model architecture from your local repository
from model.geopixel import GeoPixelForCausalLM 

# ==========================================
# 1. Configuration (Update these paths!)
# ==========================================
BASE_MODEL = "MBZUAI/GeoPixel-7B-RES" # Set to "MBZUAI/GeoPixel-7B" if you didn't do the RES task
LORA_PATH = "/content/drive/MyDrive/Capstone/GeoPixel/checkpoint-last"
TEST_IMAGE_PATH = "/content/drive/MyDrive/Capstone/dataset/test/reference/GL296_169.png" 
TEST_PROMPT = """
Segment the TARGET in this image. 
TARGET is an administrative[1] and municipal[7] district (raion), one of the fifty-four in the RED, GREEN. It is located in the southwest of the republic and borders BLUE in the north, YELLOW in the northeast, MAGENTA in the east, CYAN in the south, ORANGE in the southwest, and PURPLE in the west. The area of the district is 3,371 square kilometers (1,302 sq mi).[2] Its administrative center is the rural locality (a selo) of PINK.[3] As of the 2010 Census, the total population of the district was 31,444, with the population of PINK accounting for 27.6% of that number.[4]
"""

# ==========================================
# 2. Tokenizer Setup (Matches your train.py)
# ==========================================
print("Loading Tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    BASE_MODEL,
    padding_side='right',
    use_fast=False,
    trust_remote_code=True,
)
# We MUST add these, or the LoRA adapter will have a tensor size mismatch!
special_tokens = ['[SEG]', '<p>', '</p>']
tokenizer.add_tokens(special_tokens, special_tokens=True)

# ==========================================
# 3. Base Model Initialization
# ==========================================
print("Loading Base GeoPixel Model (This takes a minute)...")
base_model = GeoPixelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda()

# Resize the base model's embeddings to accommodate the 3 special tokens
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.tokenizer = tokenizer

# ==========================================
# 4. Inject LoRA Adapter
# ==========================================
print("Injecting Custom LoRA Weights...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("Model Ready!")

# ==========================================
# 5. Run Inference
# ==========================================
print(f"Processing image: {TEST_IMAGE_PATH}")
image = Image.open(TEST_IMAGE_PATH).convert("RGB")

# GeoPixel (via InternLM-XComposer) typically uses a built-in chat/generate wrapper.
# If your repo has a specific infer.py, mimic its generation call. Otherwise, this is standard:
with torch.no_grad():
    response = model.chat(
        tokenizer=tokenizer,
        query=TEST_PROMPT,
        image=image,
        history=[]
    )

print("\n=== MODEL OUTPUT ===")
print(response)