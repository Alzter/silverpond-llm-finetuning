#!/usr/bin/env python
# coding: utf-8

import argparse
import os

def main(args):
    
    print(args.cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

    # Load model
    device_map = "cuda:0" if len(args.cuda_visible_devices) == 1 else "auto"
    model, tokenizer = ft.load_llm(args.model_name_or_path, quantized=args.model_quantized, device_map=device_map)
    
    # TODO: 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("technique_name", type=str, help='The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".')
    parser.add_argument("model_name_or_path", type=str, help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument("max_tokens", type=int, help='How many tokens the LLM is allowed to produce to classify each sample.')
    
    parser.add_argument("dataset_name_or_path", type=str, help='Which training dataset to use. Can be a dataset from the HuggingFace Hub or the path of a CSV file to load.')
    parser.add_argument("text_columns", type=str, help='Which column(s) to use from the dataset as input text (X).')
    parser.add_argument("label_columns", type=str, help='Which column(s) to use from the dataset as output labels (y).')
    
    parser.add_argument("-c", "--cuda_visible_devices", type=str, default="1", help='Which GPU devices to use to evaluate the model. For multiple devices, separate values with commas.')
    parser.add_argument("-o", "--out_path", type=str, default='results', help="Which path to save evaluation results to.")
    
    parser.add_argument("-mq", "--model_quantized", action="store_true", help='Load the LLM with 4-bit quantization.')
    parser.add_argument("--epochs", type=int, default=2, help="How many epochs to run training for")
    parser.add_argument("--lora_rank_dimension", type=float, default=6, help="The rank of the adapter, the lower the fewer parameters you'll need to train. (smaller = more compression)")
    parser.add_argument("--lora_alpha", type=float, default=8, help="The scaling factor for LoRA layers (higher = stronger adaptation)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers (helps prevent overfitting)")
    parser.add_argument("--max_seq_length", type=int, default=64, help="")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Controls how strongly loss should affect the LoRA adapters")
	

    args = parser.parse_args()

    main(args)

