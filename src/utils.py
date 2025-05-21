import os
import warnings
from abc import ABC, abstractmethod
import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
import packaging.version

import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import time

import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer

from litellm.utils import get_llm_provider

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

@dataclass
class LocalModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cuda_devices: str = field(
        default="1",
        metadata={"help":"Comma-separated list of GPU IDs to use for training."}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(
        metadata={"help":'Scaling factor for LoRA layers (higher = stronger adaptation)'},
        default=16
    )
    lora_dropout: Optional[float] = field(
        metadata={"help":'Dropout probability for LoRA layers (helps prevent overfitting)'},
        default=0.1
    )
    lora_r: Optional[int] = field(
        metadata={"help":'Rank of the LoRA adapter. Lower ranks train fewer parameters (smaller = more compression)'},
        default=64
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    attn_implementation : Optional[str] = field(
        default="sdpa",
        metadata={"help":'The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual "eager" implementation.'}
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
    default=False,
        metadata={"help": "Enables UnSloth for training."},
    )

@dataclass
class CloudModelArguments:
	cloud_model_name : Optional[str] = field(
        metadata = {"help" : 'If provided, loads an LLM from the cloud using LiteLLM rather than locally. Requires an API key depending on model vendor.'},
		default = None
    )
	openai_api_key : Optional[str] = field(
		metadata={"help" : "Access key for OpenAI LLMs."},
		default = None
	)
	anthropic_api_key : Optional[str] = field(
		metadata={"help" : "Access key for Anthropic LLMs."},
		default = None
	)
	huggingface_api_key : Optional[str] = field(
		metadata={"help" : "Access key for HuggingFace LLMs."},
		default = None
	)
	azure_api_key : Optional[str] = field(
		metadata={"help" : "Access key for Microsoft Azure LLMs."},
		default = None
	)
	azure_api_base : Optional[str] = field(
		metadata={"help" : "Access key for Microsoft Azure LLMs."},
		default = None
	)
	azure_api_version : Optional[str] = field(
		metadata={"help" : "Access key for Microsoft Azure LLMs."},
		default = None
	)

@dataclass
class ModelResponse():
    text : str = field(metadata={"help":"Raw message content from LLM output."})
    prompt_tokens : int = field(metadata={"help":"Number of tokens used by the LLM to read the prompt."})
    completion_tokens : int = field(metadata={"help":"Number of tokens used by the LLM to answer the prompt."})
    total_tokens : int = field(metadata={"help":"Number of tokens used by the LLM to answer the query."})
    latency : float = field(metadata={"help":"How long an LLM response took to generate in seconds."})
    reasoning : Optional[str] = field(
        default=None,
        metadata={"help":"The internal reasoning generated by the LLM before writing the response."}
        )
    exception : Optional[Exception] = field(
        default=None,
        metadata={"help":"Error traceback if model generation fails."}
    )

    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})

    @classmethod
    def from_json(cls, json_file : str):
        with open(json_file) as f:
            data = json.load(f)
            f.close()
        return cls.from_dict(data) 

    def to_dict(self): return asdict(self)
    def save_json(self, path : str):
        data = self.to_dict()
        with open( path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

class PretrainedLM(ABC):
 
    @abstractmethod
    def generate(
        self,
        prompt : str | list,
        max_new_tokens : int = 64,
        temperature : float = 0,
        top_p : float | None = None,
        kwargs : dict = {}
        ) -> ModelResponse:
        """
        Generate an LLM response to a given query.

        Args:
            prompt (str | list): The prompt for the LLM.
                                You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                                if you want to include a system prompt and/or prior chat history.
            max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
            temperature (float, optional): Sampling temperature to be used. Higher = greater likelihood of low probability words. Defaults to 0.
            top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if temperature > 0. Defaults to None.
            kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
        
        Returns:
            response (str): The LLM's response.
        """
        raise NotImplementedError()

class LocalPLM(PretrainedLM):
    def __init__(self, args : LocalModelArguments, training_args : SFTConfig | None = None):

        from transformers import set_seed
        set_seed(42) # Enable deterministic LLM output

        if args.use_unsloth:
            from unsloth import FastLanguageModel
        bnb_config = None
        quant_storage_dtype = None
    
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
            and args.use_unsloth
        ):
            raise NotImplementedError("Unsloth is not supported in distributed training")
    
        if args.use_4bit_quantization:
            compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
            quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)
    
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit_quantization,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.use_nested_quant,
                bnb_4bit_quant_storage=quant_storage_dtype,
            )
    
            if compute_dtype == torch.float16 and args.use_4bit_quantization:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    print("=" * 80)
                    print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                    print("=" * 80)
            elif args.use_8bit_quantization:
                bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)
    
        if args.use_unsloth:
            # Load model
            model, _ = FastLanguageModel.from_pretrained(
                model_name=args.model_name_or_path,
                max_seq_length=training_args.max_seq_length,
                dtype=None,
                load_in_4bit=args.use_4bit_quantization,
            )
        else:
            torch_dtype = (
                quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                attn_implementation=args.attn_implementation,#"flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )
    
        peft_config = None
        chat_template = None
        if args.use_peft_lora:# and not args.use_unsloth:
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.lora_target_modules.split(",")
                if args.lora_target_modules is not None and args.lora_target_modules != "all-linear"
                else args.lora_target_modules,
            )
    
        special_tokens = None
        chat_template = None
        if args.chat_template_format == "chatml":
            special_tokens = ChatmlSpecialTokens
            chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
        elif args.chat_template_format == "zephyr":
            special_tokens = ZephyrSpecialTokens
            chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE
    
        if special_tokens is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                pad_token=special_tokens.pad_token.value,
                bos_token=special_tokens.bos_token.value,
                eos_token=special_tokens.eos_token.value,
                additional_special_tokens=special_tokens.list(),
                trust_remote_code=True,
            )
            tokenizer.chat_template = chat_template
    
            # make embedding resizing configurable?
            # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
            # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
            # ante). See https://github.com/huggingface/accelerate/issues/1620.
            uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
            uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
            if (bnb_config is not None) and uses_fsdp and uses_transformers_4_46:
                model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
            else:
                model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
    
        if args.use_unsloth:
            # Do model patching and add fast LoRA weights
            model = FastLanguageModel.get_peft_model(
                model,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                target_modules=args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
                random_state=training_args.seed,
                max_seq_length=training_args.max_seq_length,
            )
        
        self.model, self.peft_config, self.tokenizer = model, peft_config, tokenizer
    
    def _format_prompt(self, prompt : str | list[str,str]) -> str:
        """
        Convert an LLM prompt into string format with a chat template
        and special tokens.

        Args:
            prompt (str | dict[str,str]): The prompt for the LLM.
                                You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                                if you want to include a system prompt and/or prior chat history.
            tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.

        Returns:
            prompt (str): The prompt with chat template applied converted to string format using special tokens.
        """
        if type(prompt) is str:
            prompt = [{"role": "user", "content": prompt}]
        
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def generate(
        self,
        prompt : str | list,
        max_new_tokens : int = 64,
        temperature : float = 0,
        top_p : float | None = None,
        kwargs : dict = {}
        ) -> ModelResponse:
        """
        Generate an LLM response to a given query.

        Args:
            prompt (str | list): The prompt for the LLM.
                                You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                                if you want to include a system prompt and/or prior chat history.
            max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
            temperature (float, optional): Sampling temperature to be used. Higher = greater likelihood of low probability words. Defaults to 0.
            top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if temperature > 0. Defaults to None.
            kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
        
        Returns:
            response (str): The LLM's response.
        """
        time_started = time.time()

        # Convert user query into a formatted prompt
        prompt = self._format_prompt(prompt)

        # Tokenize the formatted prompt
        tokenized_input = self.tokenizer(prompt,
                                    add_special_tokens=False,
                                    return_tensors="pt").to(self.model.device)
        self.model.eval()
        
        # Set do_sample to True to enable deterministic
        # generation when temperature == 0
        do_sample = temperature == 0
        if do_sample: temperature = None

        # Generate the response
        generation_output = self.model.generate(**tokenized_input,
                                           max_new_tokens=max_new_tokens,
                                           do_sample=do_sample,
                                           temperature=temperature,
                                           top_p = top_p,
                                           #top_k = top_k,
                                           **kwargs)
        
        # Remove the tokens belonging to the prompt
        input_length = tokenized_input['input_ids'].shape[1]

        generation_output = generation_output[:, input_length:]

        output_length = generation_output.shape[1]
        
        # Decode the tokens back into text
        output = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        return ModelResponse(
            text=output,
            prompt_tokens=input_length,
            completion_tokens=output_length,
            total_tokens=input_length+output_length,
            latency=time.time() - time_started
        )
    
    def finetune(self,
        train_dataset : Dataset,
        sft_config : SFTConfig,
        lora_config : LoraConfig | None = None,
        output_dir : str | None = None,
        eval_dataset : Dataset | None = None,
        checkpoint : bool | str | None = None
        ) -> tuple[SFTTrainer, DataFrame]:
        """Fine-tune an LLM using LoRA and save the resulting adapters in ``output_dir``. The LLM specified in ``model`` **will** be modified by this function.
    
        Args:
            model (AutoModelForCausalLM): The LLM to fine-tune, which will be modified by this function. Use ``LocalPLM(LocalModelArguments)`` to instantiate.
            train_dataset (Dataset): The dataset of training samples to fine-tune the model on. You must pre-process this dataset using ``preprocess_dataset``.
            sft_config (SFTConfig): Fine-tuning training configuration, including number of epochs, checkpoints, etc.
            lora_config (LoraConfig, optional): LoRA fine-tuning hyperparameters to use. If not given, defaults to ``self.peft_config``. Defaults to None.
            output_dir (str, optional): Where to save the fine-tuned model to. Defaults to ``sft_config.output_dir``. Defaults to None.
            eval_dataset (Dataset, optional): The dataset of training samples to validate the model on. You must pre-process this dataset using ``preprocess_dataset``. Defaults to None.
            checkpoint (bool | str, optional): If present, training will resume from the model/optimizer/scheduler states loaded here. If a ``str``, local path to a saved checkpoint as saved by a previous instance of Trainer. If a bool and equals ``True``, load the last checkpoint in ``args.output_dir`` as saved by a previous instance of Trainer.
    
        Returns:
            trainer (SFTTrainer): The SFTTrainer used to fine-tune the LLM.
            result (DataFrame): The training history as a DataFrame. The columns are ["step", "loss"], where "step" is the epoch.
        """
        
        if type(self.model) is AutoPeftModelForCausalLM:
            raise Exception("Cannot finetune model because it is already finetuned. Merge the adapters into base model to train further.")
        
        if output_dir is None:
            output_dir = sft_config.output_dir
    
        if not lora_config:
            if self.peft_config: lora_config = self.peft_config
            else: raise ValueError("No LoraConfig was provided to the model for finetuning.")

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
    
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
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
    
    def save_training_history(self, history : DataFrame, output_dir : str):
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

class CloudPLM(PretrainedLM):
    def __init__(self, args : CloudModelArguments):
        
        # Raise an exception if model_name is
        # not a model that LiteLLM supports
        get_llm_provider(args.cloud_model_name)
        
        self.model = args.cloud_model_name
        
        # Set up API keys for model usage
        if args.openai_api_key: os.environ["OPENAI_API_KEY"] = args.openai_api_key
        if args.anthropic_api_key: os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
        if args.huggingface_api_key: os.environ["HUGGINGFACE_API_KEY"] = args.huggingface_api_key
        if args.azure_api_key: os.environ["AZURE_API_KEY"] = args.azure_api_key
        if args.azure_api_base: os.environ["AZURE_API_BASE"] = args.azure_api_base
        if args.azure_api_version: os.environ["AZURE_API_VERSION"] = args.azure_api_version

    def generate(
        self,
        prompt : str | list,
        max_new_tokens : int = 64,
        temperature : float = 0,
        top_p : float | None = None,
        kwargs : dict = {}
        ) -> ModelResponse:
        """
        Generate an LLM response to a given query.

        Args:
            prompt (str | list): The prompt for the LLM.
                                You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                                if you want to include a system prompt and/or prior chat history.
            max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
            temperature (float, optional): Sampling temperature to be used. Higher = greater likelihood of low probability words. Defaults to 0.
            top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if temperature > 0. Defaults to None.
            kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
        
        Returns:
            response (str): The LLM's response.
        """

        time_started = time.time()

        from litellm import completion
        
        if type(prompt) is str:
            prompt = [{"role": "user", "content": prompt}]

        if not type(prompt) is list: raise ValueError("Prompt must be a str or list[dict[str,str]] using chat template format (see https://huggingface.co/docs/transformers/main/en/chat_templating).")
        
        try:
            response = completion(
                model=self.model,
                messages=prompt,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_new_tokens,
                **kwargs,
                drop_params=True
            )
        except Exception as e:
            warnings.warn(f"Error generating model response with LiteLLM. Traceback: {str(e)}")
            # Bork catcher
            return ModelResponse(
                text="",
                prompt_tokens=0,completion_tokens=0,total_tokens=0,
                latency = time.time() - time_started,
                exception=e
            )
        
        try:
            message = response.choices[0].message
        except Exception as e:
            raise BadRequestError(f"Error generating model response. Traceback: {str(e)}")
        
        response_text = message.content
        reasoning_text = message.get("reasoning_content") # None if no reasoning
        response_text = response_text.strip()
        if reasoning_text: reasoning_text = reasoning_text.strip()
        
        return ModelResponse(
            text = response_text,
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            latency = time.time() - time_started,
            reasoning=reasoning_text
        )

@dataclass
class DatasetArguments:
    dataset : str = field(
		metadata = {"help" : 'Which dataset to use. Can be a dataset from the HuggingFace Hub or the path of a CSV file to load.'}
	)
    text_columns : str = field(
		metadata = {"help" : 'Which column(s) to use from the dataset as input text (X).'}
	)
    label_columns : str = field(
		metadata = {"help" : 'Which column(s) to use from the dataset as output labels (y).'}
    )
    test_size : float = field(
        default = 0,
        metadata = {"help" : "What percentage ratio of the dataset should be reserved for testing."}
    )
    ratio : Optional[float] = field(
        default = None,
        metadata = {"help" : "Fraction of the dataset to sample randomly ∈ (0, 1]. Cannot be used with size."}
    )
    size : Optional[int] = field(
        default = None,
        metadata = {"help" : "Number of items to sample randomly from the dataset. Cannot be used with ratio."}
    )
    # dataset_name: Optional[str] = field(
    #     default="timdettmers/openassistant-guanaco",
    #     metadata={"help": "The preference dataset to use."},
    # )
    # append_concat_token: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    # )
    # add_special_tokens: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    # )
    # splits: Optional[str] = field(
    #     default="train,test",
    #     metadata={"help": "Comma separate list of the splits to use from the dataset."},
    # )

class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


# def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
#     def preprocess(samples):
#         batch = []
#         for conversation in samples["messages"]:
#             batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
#         return {"content": batch}
# 
#     raw_datasets = DatasetDict()
#     for split in data_args.splits.split(","):
#         try:
#             # Try first if dataset on a Hub repo
#             dataset = load_dataset(data_args.dataset_name, split=split)
#         except DatasetGenerationError:
#             # If not, check local dataset
#             dataset = load_from_disk(os.path.join(data_args.dataset_name, split))
# 
#         if "train" in split:
#             raw_datasets["train"] = dataset
#         elif "test" in split:
#             raw_datasets["test"] = dataset
#         else:
#             raise ValueError(f"Split type {split} not recognized as one of test or train.")
# 
#     if apply_chat_template:
#         raw_datasets = raw_datasets.map(
#             preprocess,
#             batched=True,
#             remove_columns=raw_datasets["train"].column_names,
#         )
# 
#     train_data = raw_datasets["train"]
#     valid_data = raw_datasets["test"]
#     print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
#     print(f"A sample of train dataset: {train_data[0]}")
# 
#     return train_data, valid_data


# def create_and_prepare_model(args : LocalModelArguments):#, training_args):
#     # if args.use_unsloth:
#     #     from unsloth import FastLanguageModel
#     bnb_config = None
#     quant_storage_dtype = None

#     # if (
#     #     torch.distributed.is_available()
#     #     and torch.distributed.is_initialized()
#     #     and torch.distributed.get_world_size() > 1
#     #     and args.use_unsloth
#     # ):
#     #     raise NotImplementedError("Unsloth is not supported in distributed training")

#     if args.use_4bit_quantization:
#         compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
#         quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=args.use_4bit_quantization,
#             bnb_4bit_quant_type=args.bnb_4bit_quant_type,
#             bnb_4bit_compute_dtype=compute_dtype,
#             bnb_4bit_use_double_quant=args.use_nested_quant,
#             bnb_4bit_quant_storage=quant_storage_dtype,
#         )

#         if compute_dtype == torch.float16 and args.use_4bit_quantization:
#             major, _ = torch.cuda.get_device_capability()
#             if major >= 8:
#                 print("=" * 80)
#                 print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
#                 print("=" * 80)
#         elif args.use_8bit_quantization:
#             bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

#     # if args.use_unsloth:
#     #     # Load model
#     #     model, _ = FastLanguageModel.from_pretrained(
#     #         model_name=args.model_name_or_path,
#     #         max_seq_length=training_args.max_seq_length,
#     #         dtype=None,
#     #         load_in_4bit=args.use_4bit_quantization,
#     #     )
#     #else:
#     if True:
#         torch_dtype = (
#             quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_name_or_path,
#             quantization_config=bnb_config,
#             trust_remote_code=True,
#             attn_implementation=args.attn_implementation,#"flash_attention_2" if args.use_flash_attn else "eager",
#             torch_dtype=torch_dtype,
#         )

#     peft_config = None
#     chat_template = None
#     if args.use_peft_lora:# and not args.use_unsloth:
#         peft_config = LoraConfig(
#             lora_alpha=args.lora_alpha,
#             lora_dropout=args.lora_dropout,
#             r=args.lora_r,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules=args.lora_target_modules.split(",")
#             if args.lora_target_modules is not None and args.lora_target_modules != "all-linear"
#             else args.lora_target_modules,
#         )

#     special_tokens = None
#     chat_template = None
#     if args.chat_template_format == "chatml":
#         special_tokens = ChatmlSpecialTokens
#         chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
#     elif args.chat_template_format == "zephyr":
#         special_tokens = ZephyrSpecialTokens
#         chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

#     if special_tokens is not None:
#         tokenizer = AutoTokenizer.from_pretrained(
#             args.model_name_or_path,
#             pad_token=special_tokens.pad_token.value,
#             bos_token=special_tokens.bos_token.value,
#             eos_token=special_tokens.eos_token.value,
#             additional_special_tokens=special_tokens.list(),
#             trust_remote_code=True,
#         )
#         tokenizer.chat_template = chat_template

#         # make embedding resizing configurable?
#         # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
#         # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
#         # ante). See https://github.com/huggingface/accelerate/issues/1620.
#         uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
#         uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
#         if (bnb_config is not None) and uses_fsdp and uses_transformers_4_46:
#             model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
#         else:
#             model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
#         tokenizer.pad_token = tokenizer.eos_token

#     # if args.use_unsloth:
#     #     # Do model patching and add fast LoRA weights
#     #     model = FastLanguageModel.get_peft_model(
#     #         model,
#     #         lora_alpha=args.lora_alpha,
#     #         lora_dropout=args.lora_dropout,
#     #         r=args.lora_r,
#     #         target_modules=args.lora_target_modules.split(",")
#     #         if args.lora_target_modules != "all-linear"
#     #         else args.lora_target_modules,
#     #         use_gradient_checkpointing=training_args.gradient_checkpointing,
#     #         random_state=training_args.seed,
#     #         max_seq_length=training_args.max_seq_length,
#     #     )

#     return model, peft_config, tokenizer
