import os
import sys
import torch
import argparse
import transformers
from peft import PeftModel
from transformers import AutoTokenizer

from model.geopixel import GeoPixelForCausalLM


def parse_args(args):
    parser = argparse.ArgumentParser(description="merge lora weights and save model with hf format")
    parser.add_argument("--version", default="MBZUAI/GeoPixel-7B")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default='facebook/sam2-hiera-large', type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="GeoPixel-7B", type=str)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    added_tokens = [
        '<p>', '</p>', '<unused_1>', '<unused_2>',
        '<unused_3>', '<unused_4>', '[SEG]', '<unused_5>', '<unused_6>'
    ]

    tokenizer.add_tokens(added_tokens)

    tokenizer.pad_token = tokenizer.unk_token

    args.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
    args.bop_token_idx = tokenizer.convert_tokens_to_ids('<p>')
    args.eop_token_idx = tokenizer.convert_tokens_to_ids('</p>')

    model_args = {
        "vision_pretrained": args.vision_pretrained,
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "bop_token_idx": args.bop_token_idx,
        "eop_token_idx": args.eop_token_idx,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    print(f"Loading base model from: {args.version} ...")
    model = GeoPixelForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.model.initialize_geopixel_modules(model.model.config)
    model.resize_token_embeddings(len(tokenizer))

    # adapter_weight can be either a directory (containing adapter_config.json)
    # or a direct path to adapter_model.bin / adapter_model.safetensors
    adapter_path = args.weight
    if os.path.isfile(adapter_path):
        adapter_path = os.path.dirname(adapter_path)

    print(f"Loading LoRA adapter from: {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch_dtype)

    print("Merging adapter layers ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.save_path} ...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print("Model merged and saved successfully.")

if __name__ == "__main__":
    main(sys.argv[1:])
