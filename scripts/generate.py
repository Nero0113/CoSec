import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, default='human-eval-starcode-1b-t08')
    parser.add_argument('--model_name_or_path', type=str, default='../starcode-1b')
    parser.add_argument('--base_model', type=str, default='../starcode-1b')
    parser.add_argument('--sec_model', type=str, default='../trained/starcode/sec/checkpoint-last')

    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--num_samples_per_gen', type=int, default=25)
    parser.add_argument('--exp_temp', type=float, default=0.4)

    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'deepseek' in args.model_name_or_path:
        model = CodeLlamaModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'star' in args.model_name_or_path:
        model = StarcodeModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'codegen' in args.model_name_or_path:
        model = CodegenModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)

    with torch.no_grad():
