#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_name_or_path", type="str", "Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("prompt", type="str", "Evaluation Prompt to ")

# # ▶️ Initial Setup
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
ROOT_DIR = "../data/eq/"

import os

# Import modules for LLM finetuning and evaluation
import finetune as ft
import evaluate as ev

from evaluate import EvaluationConfig

def evaluate_model(configurations, model, tokenizer, label_names, eval_dataset):
    results = []
    for config in configurations:
        result = ev.evaluate(
            model=model, tokenizer=tokenizer, label_names=label_names,
            eval_dataset=eval_dataset, eval_config=config
        )
        results.append(result)
    return results

def save_results(results):
    for result in results:
        result.save(os.path.join(ROOT_DIR, "results")) # Saves to "data/eq/results/<EvaluationConfig.name>"


# # ▶️ Load EQ dataset
import pandas as pd

data_train = pd.read_csv( os.path.join(ROOT_DIR, "datasets/preprocessed_train.csv"), low_memory=False )
data_eval  = pd.read_csv( os.path.join(ROOT_DIR, "datasets/preprocessed_eval.csv"),  low_memory=False )

input_features = [
"OUTAGE_ID",
"WEATHER_CONDITION",
"OUTAGE_CAUSE",
"FAULT_LONG_DESCRIPTION",
"SHORT_DESC_2",
"WORK_ORDER_COMPONENT_CODE_DESCRIPTION",
"OUTAGE_CAUSE_GROUP",
"OUTAGE_STANDARD_REASON_DESCRIPTION",
"REASON_FOR_INTERRUPTION",
"PROVIDER"
]

output_labels = [
    "MSSS_OBJECT_DESCRIPTION",
    "MSSS_DAMAGE_DESCRIPTION",
    "MSSS_CAUSE_DESCRIPTION"
]

# Convert DataFrame into a Dataset
dataset = ft.create_dataset_from_dataframe(data_train, input_features, output_labels, test_size=0.2)

# Reduce the size of the dataset for testing purposes
# dataset['train'] = dataset['train'].shard(5, 0) # First 20% of the train dataset
# dataset['test'] = dataset['test'].shard(5, 0) # First 20% of the test dataset

# Preprocess the dataset into a form usable for supervised finetuning
dataset, label_names = ft.preprocess_dataset(dataset,text_columns=input_features,label_columns=output_labels)


# # ▶️ Load Baseline LLM
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DEVICE = "cuda:0"
QUANTIZED = True # Load model with 4-bit quantization

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Same quantization configuration as QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.float16
) if QUANTIZED else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=MODEL_DEVICE,
    use_cache=False # use_cache is incompatible with gradient checkpointing
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# # ▶️ Evaluate Baseline LLM
labels = list(label_names.keys())
values = list(label_names.values())
PROMPT = f"""You are an expert at classifying power outage reports.

You are given three classification tasks.
You should output the result as a two json fields as {{"MSSS_OBJECT_DESCRIPTION" : "object_label", "MSSS_DAMAGE_DESCRIPTION" : "damage_label", "MSSS_CAUSE_DESCRIPTION" : "cause_label"}}
For {labels[0]}, given the outage report, you are asked to classify it as one of the labels in the list {values[0]} and change object_label to the correct label in the list.
For {labels[1]}, given the outage report, you are asked to classify it as one of the labels in the list {values[1]} and change damage_label to the correct label in the list.
For {labels[2]}, given the outage report, you are asked to classify it as one of the labels in the list {values[2]} and change cause_label to the correct label in the list.

Output the three json fields only and absolutely nothing else.
Now it is your turn."""

data

# Configurations for the baseline LLM
baseline_configurations = [
    EvaluationConfig(
        name="Zero-shot Multi-task",
        prompt=PROMPT,
        max_tokens=50
    )
]

baseline_results = evaluate_model(baseline_configurations, model=model, tokenizer=tokenizer, label_names=label_names, eval_dataset=dataset['test'])

save_results(baseline_results)


# # ▶️ Finetune LLM
FINETUNED_LLM_PATH = os.path.join(ROOT_DIR, "models/Qwen2.5-FT-EQ-4")

LORA_RANK_DIMENSION = 6 # the rank of the adapter, the lower the fewer parameters you'll need to train. (smaller = more compression)
LORA_ALPHA = 8 # this is the scaling factor for LoRA layers (higher = stronger adaptation)
LORA_DROPOUT = 0.05 # dropout probability for LoRA layers (helps prevent overfitting)
MAX_SEQ_LENGTH = 64
EPOCHS=2
LEARNING_RATE=2e-4

from peft import LoraConfig

lora_config = LoraConfig(
    r=LORA_RANK_DIMENSION,
    lora_alpha=LORA_ALPHA,
    bias="none",
    lora_dropout=LORA_DROPOUT,
    task_type="CAUSAL_LM"
)

from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=1,
    per_device_train_batch_size=16,
    auto_find_batch_size=True,

    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,

    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    optim='adamw_torch_fused',
    warmup_ratio=0.03,
    lr_scheduler_type="constant", 

    logging_steps=10,
    logging_dir='./logs',
    output_dir=FINETUNED_LLM_PATH,
    report_to='none'
)

history = ft.finetune( # Will save the model to the directory: FINETUNED_LLM_PATH
    model=model, tokenizer=tokenizer,
    train_dataset=dataset['train'],
    lora_config=lora_config, sft_config=sft_config
)

history.to_csv(os.path.join(ROOT_DIR, "results/loss_history.csv"), index=False) # Save the training history

history

# Plot the training history and save the plot
import math
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.plot(history.set_index("step")["loss"])
plt.xlabel("Epoch")
plt.ylabel("Training Loss")

loss_max = math.ceil(history['loss'].max())
plt.ylim([0, loss_max])

plt.title("Fine-tuning Training History")

path = os.path.join(ROOT_DIR, "results/loss_history.png")
plt.savefig( path, dpi=200, bbox_inches='tight' )


# # ▶️ Load Finetuned LLM
# Unload the baseline model if it exists, otherwise we will probably get an OOM exception
import gc, torch

if "bnb_config" in locals(): del bnb_config
if "tokenizer" in locals(): del tokenizer
if "model" in locals(): del model
gc.collect()
torch.cuda.empty_cache()

# FINETUNED_LLM_PATH = os.path.join(ROOT_DIR, "models/Qwen2.5-FT-EQ-1")
MODEL_DEVICE = "cuda:0"
QUANTIZED = True # Load model with 4-bit quantization

model, tokenizer = ft.load_finetuned_llm(FINETUNED_LLM_PATH, MODEL_DEVICE, QUANTIZED)


# # ▶️ Evaluate Finetuned LLM
finetuned_configurations = [
    EvaluationConfig(
        name="Fine-tuned Multi-task",
        prompt=None,
        max_tokens=50
    )
]

finetuned_results = evaluate_model(finetuned_configurations, model=model, tokenizer=tokenizer, label_names=label_names, eval_dataset=dataset['test'])

save_results(finetuned_results)



