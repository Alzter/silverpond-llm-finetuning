#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("technique_name", type=str, help='The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".')
parser.add_argument("model_name_or_path", type=str, help='Path to pretrained model or model identifier from huggingface.co/models.')
parser.add_argument("max_tokens", type=int, help='How many tokens the LLM is allowed to produce to classify each sample.')

parser.add_argument("dataset", type=str, help="Path to dataset to evaluate model from.")


parser.add_argument("-o", "--out_path", type=str, default='results', help="Path to save evaluation results.")

parser.add_argument("--prompt", type=str, help='Optional prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.')
parser.add_argument("--prompt_role", type=str, default='system', help='What role to give the LLM prompt. Defaults to "system", meaning a system prompt. Can be replaced with "user" for models which do not work well with system prompts.')
parser.add_argument("--do_sample", action="store_true", help='If False, enables deterministic generation.')
parser.add_argument("--temperature", type=float, help='Higher = greater likelihood of low probability words. Leave empty if do_sample is False.')
parser.add_argument("--top_p", type=float, help='If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Leave empty if do_sample is False.')
parser.add_argument("--top_k", type=float, help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Leave empty if do_sample is False.')

args = parser.parse_args()

print(f"Name: {args.technique_name}")
print(f"Model: {args.model_name_or_path}")
print(f"Tokens: {args.max_tokens}")
print(f"Prompt: {args.prompt}")
print(f"Prompt is None: {args.prompt is None}")
print(f"Prompt Role: {args.prompt_role}")

import evaluate
from evaluate import EvaluationConfig
config = EvaluationConfig(
    name = args.technique_name,
    max_tokens = args.max_tokens,
    prompt = args.prompt,
    prompt_role = args.prompt_role,
    do_sample = args.do_sample,
    temperature = args.temperature,
    top_p = args.top_p,
    top_k = args.top_k
)

# TODO: Load dataset

# TODO: Load model

# TODO: Run evaluation

# TODO: Save evaluation results
