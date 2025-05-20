import pytest
import evaluate as ev
import preprocess as pre
from utils import LocalPLM, LocalModelArguments
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch
import pandas as pd

@pytest.fixture
def model():
    args = LocalModelArguments(
        model_name_or_path = "Qwen/Qwen2.5-7B-Instruct",
        use_4bit_quantization = True,
        bnb_4bit_quant_type = "nf4", # QLoRA uses 4-bit NormalFloat precision,
        use_nested_quant = True, # QLoRA uses double quantizing
        bnb_4bit_compute_dtype = "float16",
        cuda_devices = "3"
    )
    
    model = LocalPLM(args)
    return model 

@pytest.fixture
def eval_dataset():
    eval_data = load_dataset("fancyzhx/dbpedia_14", split="test")
    eval_data = pre.undersample_dataset(eval_data, label_columns="label", samples_per_class=2)
    eval_data, label_names = pre.preprocess_dataset(eval_data, text_columns="content", label_columns="label")
    return (eval_data, label_names)

@pytest.fixture()
def csv_dataset():
    return pd.read_csv("tests/test_data.csv", low_memory=False)

@pytest.fixture()
def eval_prompt(eval_dataset):
    _, label_names = eval_dataset

    return ev.create_prompt(
        data_sample_name = "article",
        label_names = label_names
    )

def test_create_prompt(eval_prompt):

    expected = """You are an expert at classifying articles into the following categories:

CATEGORIES:
0. Company
1. EducationalInstitution
2. Artist
3. Athlete
4. OfficeHolder
5. MeanOfTransportation
6. Building
7. NaturalPlace
8. Village
9. Animal
10. Plant
11. Album
12. Film
13. WrittenWork

Read the following article then answer with the name of the category which suits it best.
Answer with ONLY the name of the category, i.e. "Company"."""
    
    assert(expected == eval_prompt, "Generating an evaluation prompt should follow a specific structure.")

def test_evaluate_llm(model, eval_dataset, eval_prompt):
    eval_data, label_names = eval_dataset

    eval_config = ev.EvaluationConfig(
        technique_name="Zero-shot",
        prompt = eval_prompt,
        max_tokens = 3
    )

    result = ev.evaluate(
        model,
        label_names=label_names,
        eval_dataset=eval_data,
        eval_config=eval_config
    )



