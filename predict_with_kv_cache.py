# inference.py


import copy
import json
import logging
import argparse
import torch
import os
import wandb
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import transformers
import numpy as np
import random

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="A simple inference script for llava-ov multimodal model.")
    # settings for path/basic
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="coco", help="")
    parser.add_argument("--data_folder", type=str, default="/share/home/mhma/datasets/after_process/sharegpt4v_coco/")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image_id", type=str, default="")
    # settings for model configuration
    
    parser.add_argument('--pretrained', type=str, default="/share/home/mhma/models/llava-onevision-qwen2-7b-ov", help='Pretrained model path or identifier.')
    parser.add_argument('--model_name', type=str, default="llava_qwen", help='Model name.')
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="")
    parser.add_argument("--device_map", type=str, default="auto", help="")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="")
    parser.add_argument("--multimodal", type=bool, default=True, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    
    
    # settings for kv cache method
    parser.add_argument('--method', type=str, choices=['streamingllm', 
                                                       'h2o', 
                                                       'snapkv', 
                                                       'look-m', 
                                                       'vl-cache', 
                                                       'adakv'], help='KV cache method to use.')
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")
    
    
    
    
    ## mhma TODO: continuously add more arguments, such as anyres/quantization/etc.
    # settings for anyres
    parser.add_argument("--anyres", type=bool, default=False, help="") 
    
          
    ## mhma TODO: consider create some classes later for args such as @dataclass ModelArgs, etc.
    return parser.parse_args()


def replace_layers(args):
    from kv_cache_compression.monkeypatch import replace_qwen,replace_mistral,replace_llama
    if "qwen" in args.model_name.lower():
        replace_qwen(args.method.lower())
    elif "mistral" in args.model_name.lower():
        replace_mistral(args.method.lower())
    elif "llama" in args.model_name.lower():
        replace_llama(args.method.lower())
    else:
        raise ValueError(f"Model name {args.model_name} not supported")


def load_model(args, pretrained, model_name):
    llava_model_args = {
        "attn_implementation": args.attn_implementation, 
        "device_map": args.device_map, 
        "torch_dtype": args.torch_dtype,
        "multimodal": args.multimodal
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args)
    
    # replace some layers impl of kv cache method
    model.eval()
    return tokenizer, model, image_processor, max_length


def run_inference(model, tokenizer, image_processor,args):
    
    device = args.device
    if args.model_name == "llava_qwen":
        conv_template = "qwen_1_5" 
    
    # following code will change after debugging(image load part)
    with open(args.data_folder+'caption.json', 'r') as f:
        text = json.load(f)    
    

    image_path = args.data_folder + 'images/'+ args.image_id+'.jpg'
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    result = next((item for item in text if item.get("id") == args.image_id), None)
    question = result['conversations'][0]['value']
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    image_sizes = [image.size]

    cont = model.generate(
        input_ids,  
        images=image_tensor,  
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    output = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(output)
    
    
    
def main():
    args = parse_args()
    tokenizer, model, image_processor, max_length = load_model(args, args.pretrained, args.model_name)
    if args.method:
        replace_layers(args)
    run_inference(model, tokenizer, image_processor, args)

if __name__ == "__main__":
    main()