import pytest
import evaluate as ev
import finetune as ft
import model_prompts
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch

@pytest.fixture
def llm():
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit = True,
    #     bnb_4bit_quant_type = "nf4", # QLoRA uses 4-bit NormalFloat precision,
    #     bnb_4bit_use_double_quant = True, # QLoRA uses double quantising,
    #     bnb_4bit_compute_dtype = torch.float16
    # )

    model_id = "HuggingFaceTB/SmolLM2-1.7B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return (model, tokenizer)

@pytest.fixture
def eval_dataset():
    eval_data = load_dataset("fancyzhx/dbpedia_14", split="test")
    eval_data = ft.sample_dataset(eval_data, samples_per_class=10)
    label_names, eval_data = ft.preprocess_dataset(load_dataset("fancyzhx/dbpedia_14", split="test"), "content", "label")
    return (label_names, eval_data)

def test_evaluate_llm(llm, eval_dataset):
    model, tokenizer = llm
    label_names, eval_data = eval_dataset

    eval_config = ev.ClassificationMethod(
        max_tokens = 100,
        prompt = model_prompts.PROMPT_ZEROSHOT
    )

    result = ev.evaluate(
        model, tokenizer,
        label_names=label_names,
        eval_dataset=eval_data,
        eval_config=eval_config
    )



