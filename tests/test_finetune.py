# import pytest
# import finetune as ft
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from datasets import load_dataset
# import torch

# @pytest.fixture
# def llm():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit = True,
#         bnb_4bit_quant_type = "nf4", # QLoRA uses 4-bit NormalFloat precision,
#         bnb_4bit_use_double_quant = True, # QLoRA uses double quantising,
#         bnb_4bit_compute_dtype = torch.float16
#     )

#     model_id = "Qwen/Qwen2.5-7B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     return (model, tokenizer)