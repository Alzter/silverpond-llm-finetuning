#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from evaluate import EvaluationConfig
from utils import ModelArguments, DatasetArguments, create_and_prepare_model
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

def main(eval_config : EvaluationConfig, model_args : ModelArguments, data_args : DatasetArguments):
    pass 
    # print(args.cuda_visible_devices)
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices
    # 
    # print(f"Name: {args.technique_name}")
    # print(f"Model: {args.model_name_or_path}")
    # print(f"Tokens: {args.max_tokens}")
    # print(f"Prompt: {args.prompt}")
    # print(f"Prompt is None: {args.prompt is None}")
    # print(f"Prompt Role: {args.prompt_role}")
    # 
    # from evaluate import EvaluationConfig
    # eval_config = EvaluationConfig(
    #     name = args.technique_name,
    #     max_tokens = args.max_tokens,
    #     prompt = args.prompt,
    #     prompt_role = args.prompt_role,
    #     do_sample = args.do_sample,
    #     temperature = args.temperature,
    #     top_p = args.top_p,
    #     top_k = args.top_k
    # )
    
    # Load evaluation dataset
    import finetune as ft
    eval_dataset, label_names = ft.load_dataset(
        data_args.dataset_name_or_path,
        data_args.text_columns,
        data_args.label_columns,
        test_size=0
    ) 

    # Load model
    model, peft_config, tokenizer = create_and_prepare_model(model_args)

    # Load model
    # device_map = "cuda:0" if len(args.cuda_visible_devices) == 1 else "auto"
    # model, tokenizer = ft.load_llm(args.model_name_or_path, quantized=args.model_quantized, device_map=device_map)
    
    # Run evaluation
    import evaluate as ev
    result = ev.evaluate(model=model,tokenizer=tokenizer,label_names=label_names,eval_dataset=eval_dataset,eval_config=eval_config)
    
    # Save evaluation results
    result.save()

if __name__ == "__main__":
    parser = HfArgumentParser((EvaluationConfig, ModelArguments, DatasetArguments))

    eval_config, model_args, data_args = parser.parse_args_into_dataclasses()

    main(args)
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument("technique_name", type=str, help='The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".')
    # parser.add_argument("model_name_or_path", type=str, help='Path to pretrained model or model identifier from huggingface.co/models.')
    # parser.add_argument("max_tokens", type=int, help='How many tokens the LLM is allowed to produce to classify each sample.')
    # 
    # parser.add_argument("dataset_name_or_path", type=str, help='Which evaluation dataset to use. Can be a dataset from the HuggingFace Hub or the path of a CSV file to load.')
    # parser.add_argument("text_columns", type=str, help='Which column(s) to use from the dataset as input text (X).')
    # parser.add_argument("label_columns", type=str, help='Which column(s) to use from the dataset as output labels (y).')
    # 
    # parser.add_argument("-c", "--cuda_visible_devices", type=str, default="1", help='Which GPU devices to use to evaluate the model. For multiple devices, separate values with commas.')
    # parser.add_argument("-o", "--out_path", type=str, default='results', help="Which path to save evaluation results to.")
    # 
    # parser.add_argument("-mq", "--model_quantized", action="store_true", help='Load the LLM with 4-bit quantization.')

    # parser.add_argument("-p", "--prompt", type=str, help='Optional prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.')
    # parser.add_argument("-pr", "--prompt_role", type=str, default='system', help='What role to give the LLM prompt. Defaults to "system", meaning a system prompt. Can be replaced with "user" for models which do not work well with system prompts.')
    # parser.add_argument("--do_sample", action="store_true", help='If False, enables deterministic generation.')
    # parser.add_argument("--temperature", type=float, help='Higher = greater likelihood of low probability words. Leave empty if do_sample is False.')
    # parser.add_argument("--top_p", type=float, help='If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Leave empty if do_sample is False.')
    # parser.add_argument("--top_k", type=float, help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Leave empty if do_sample is False.')
    
    #args = parser.parse_args()

    # main(args)

