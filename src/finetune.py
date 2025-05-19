from datasets import Dataset
from transformers import set_seed
from peft import LoraConfig, PeftConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
import pandas as pd
import torch
from pandas import DataFrame
import warnings
from matplotlib import pyplot as plt
import os
import math
from utils import LocalPLM

set_seed(42) # Enable deterministic LLM output

def finetune(
    model : LocalPLM,
    train_dataset : Dataset,
    lora_config : LoraConfig,
    sft_config : SFTConfig,
    output_dir : str | None = None,
    eval_dataset : Dataset | None = None,
    checkpoint : bool | str | None = None
    ) -> tuple[SFTTrainer, DataFrame]:
    """Fine-tune an LLM using LoRA and save the resulting adapters in ``output_dir``. The LLM specified in ``model`` **will** be modified by this function.

    Args:
        model (AutoModelForCausalLM): The LLM to fine-tune, which will be modified by this function. Use ``LocalPLM(LocalModelArguments)`` to instantiate.
        train_dataset (Dataset): The dataset of training samples to fine-tune the model on. You must pre-process this dataset using ``preprocess_dataset``.
        lora_config (LoraConfig): LoRA hyperparameters, including the rank of the adapters and the scaling factor.
        sft_config (SFTConfig): Fine-tuning training configuration, including number of epochs, checkpoints, etc.
        output_dir (str, optional): Where to save the fine-tuned model to. Defaults to ``sft_config.output_dir``. Defaults to None.
        eval_dataset (Dataset, optional): The dataset of training samples to validate the model on. You must pre-process this dataset using ``preprocess_dataset``. Defaults to None.
        checkpoint (bool | str, optional): If present, training will resume from the model/optimizer/scheduler states loaded here. If a ``str``, local path to a saved checkpoint as saved by a previous instance of Trainer. If a bool and equals ``True``, load the last checkpoint in ``args.output_dir`` as saved by a previous instance of Trainer.

    Returns:
        trainer (SFTTrainer): The SFTTrainer used to fine-tune the LLM.
        result (DataFrame): The training history as a DataFrame. The columns are ["step", "loss"], where "step" is the epoch.
    """
    
    if type(model.model) is AutoPeftModelForCausalLM:
        raise Exception("Cannot finetune model because it is already finetuned. Merge the adapters into base model to train further.")
    
    if output_dir is None:
        output_dir = sft_config.output_dir

    model.model = prepare_model_for_kbit_training(model.model)
    model.model = get_peft_model(model.model, lora_config)

    trainer = SFTTrainer(
        model=model.model,
        processing_class=model.tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model(output_dir)

    try:
        result = pd.DataFrame(trainer.state.log_history)
    except Exception as e:
        warnings.warn(f"Error saving training results for model.\n{str(e)}")
        result = None
    
    return trainer, result

def save_training_history(history : DataFrame, output_dir : str):
    """Export training history of a fine-tuned LLM as a CSV and as a plot.
    
    Args:
        history (DataFrame) : LLM fine-tuning history retrieved from finetune()
        output_dir (str) : Path to store results.
    """
    # Save the training history
    history.to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)

    # Plot the training history and save the plot
    plt.plot(history.set_index("step")["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    
    loss_max = math.ceil(history['loss'].max())
    plt.ylim([0, loss_max])
    
    plt.title("Fine-tuning Training History")
    
    path = os.path.join(output_dir, "loss_history.png")
    plt.savefig( path, dpi=200, bbox_inches='tight' )

# def load_llm(model_name_or_path : str, device_map : str = "cuda:0", quantized : bool = True) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
#     """
#     Load an LLM from disk or from huggingface.co/models.
# 
#     Args:
#         model_name_or_path (str): Name of the model.
#         device_map (str, optional): Which device to load the fine-tuned model onto. Defaults to "cuda:0".
#         quantized (bool, optional): Whether to load the model with 4-bit quantization. Defaults to True.
# 
#     Returns:
#         model (AutoPeftModelForCausalLM): The LLM.
#         tokenizer (AutoTokenizer): The tokenizer.
#     """
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=False,
#         bnb_4bit_compute_dtype=torch.float16
#     ) if quantized else None
# 
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, quantization_config=bnb_config)
# 
#     return (model, tokenizer)
# 
# def load_finetuned_llm(model_directory : str, device_map : str = "cuda:0", quantized:bool = True) -> tuple[AutoPeftModelForCausalLM, AutoTokenizer]:
#     """
#     Load a finetuned LLM from disk.
# 
#     Args:
#         model_directory (str): Where to load the fine-tuned model.
#         device_map (str, optional): Which device to load the fine-tuned model onto. Defaults to "cuda:0".
#         quantized (bool, optional): Whether to load the model with 4-bit quantization. Defaults to True.
# 
#     Returns:
#         model (AutoPeftModelForCausalLM): The fine-tuned LLM.
#         tokenizer (AutoTokenizer): The tokenizer (unchanged from the base model).
#     """
# 
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=False,
#         bnb_4bit_compute_dtype=torch.float16
#     ) if quantized else None
# 
#     config = PeftConfig.from_pretrained(model_directory)
# 
#     tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     model = AutoPeftModelForCausalLM.from_pretrained(model_directory, device_map=device_map, quantization_config=bnb_config)
# 
#     return (model, tokenizer)

def _format_prompt(prompt : str | dict, tokenizer : AutoTokenizer) -> str:
    """
    Convert an LLM prompt into string format with a chat template
    and special tokens.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.

    Returns:
        prompt (str): The prompt with chat template applied converted to string format using special tokens.
    """
    if type(prompt) is str:
        prompt = [{"role": "user", "content": prompt}]
    
    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    return prompt

def generate(
    prompt : str | dict,
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    max_new_tokens : int = 64,
    response_only : bool = True,
    skip_special_tokens : bool = True,
    do_sample : bool = False,
    temperature : float | None = None,
    top_p : float | None = None,
    top_k : float | None = None,
    kwargs : dict = {}
    ) -> str:
    """
    Generate an LLM response to a given query.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        model (AutoModelForCausalLM): The LLM to use. Use ``AutoModelForCausalLM.from_pretrained(model_name)`` to instantiate.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.
        max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
        response_only (bool, optional): If True, excludes all previous messages from the output. Defaults to True.
        skip_special_tokens (bool, optional): If True, removes model special tokens from the output. Defaults to True.
        do_sample (bool, optional): If False, enables deterministic generation. Defaults to False.
        temperature (float, optional): Higher = greater likelihood of low probability words. Leave empty if ``do_sample`` is False. Defaults to None.
        top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if ``do_sample`` is False. Defaults to None.
        top_k (float, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Leave empty if ``do_sample`` is False. Defaults to None.
        kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
    Returns:
        response (str): The LLM's response.
    """

    # Convert user query into a formatted prompt
    prompt = _format_prompt(prompt, tokenizer=tokenizer)

    # Tokenize the formatted prompt
    tokenized_input = tokenizer(prompt,
                                add_special_tokens=False,
                                return_tensors="pt").to(model.device)
    model.eval()

    # Generate the response
    generation_output = model.generate(**tokenized_input,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=do_sample,
                                       temperature=temperature,
                                       top_p = top_p,
                                       top_k = top_k,
                                       **kwargs)

    # If required, remove the tokens belonging to the prompt
    if response_only:
        input_length = tokenized_input['input_ids'].shape[1]
        generation_output = generation_output[:, input_length:]
    
    # Decode the tokens back into text
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)[0]
    return output
