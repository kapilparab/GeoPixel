import torch
import numpy as np
import transformers 
from model.geopixel import GeoPixelForCausalL

model_name = "MBZUAI/GeoPixel-7B"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    padding_side='right',
    use_fast=False,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.unk_token
seg_token_idx, bop_token_idx, eop_token_idx = [
    tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>']
]

kwargs = {"torch_dtype": torch.bfloat16}

geo_model_args = {
    "vision_pretrained": 'facebook/sam2-hiera-large',
    "seg_token_idx" : seg_token_idx, # segmentation token index
    "bop_token_idx" : bop_token_idx, # begining of phrase token index
    "eop_token_idx" : eop_token_idx  # end of phrase token index
}

model = GeoPixelForCausalLM.from_pretrained(
    model_name, 
    low_cpu_mem_usage=True, 
    **kwargs,
    **geo_model_args
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.tokenizer = tokenizer

print(f"Model vocab size: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {len(tokenizer)}")