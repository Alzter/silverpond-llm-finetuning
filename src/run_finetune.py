import os
import sys

from transformers import HfArgumentParser
from trl import SFTConfig
from utils import ModelArguments, DatasetArguments

#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def main(model_args : ModelArguments, data_args : DatasetArguments, training_args : SFTConfig):
    from transformers import set_seed
    from trl import SFTTrainer
    from utils import create_and_prepare_model   # Set seed for reproducibility
    
    set_seed(training_args.seed)

    # Load training/evaluation dataset
    import preprocess as pre
    dataset, label_names = pre.load_dataset(
        data_args.dataset,
        data_args.text_columns,
        data_args.label_columns,
        test_size = data_args.test_size
    )

    # train_dataset, eval_dataset = dataset['train'], dataset['test']

    #training_args.dataset_kwargs = {
    #    "append_concat_token": data_args.append_concat_token,
    #    "add_special_tokens": data_args.add_special_tokens,
    #}

    # datasets
    #train_dataset, eval_dataset = create_datasets(
    #    tokenizer,
    #    data_args,
    #    training_args,
    #    apply_chat_template=model_args.chat_template_format != "none",
    #)
    
    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args)
    
    import finetune as ft
    #model, tokenizer = ft.load_llm(
    #    model_args.model_name_or_path,
    #    quantized=True
    #)

    import subprocess
    subprocess.run(["nvidia-smi"])
    
    # if model_args.use_peft_lora:
    #     from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

    #     peft_config = LoraConfig(
    #         lora_alpha=model_args.lora_alpha,
    #         lora_dropout=model_args.lora_dropout,
    #         r=model_args.lora_r,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )

    #     #model = prepare_model_for_kbit_training(model)
    #     #model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing (saves memory)
    model.config.use_cache = not training_args.gradient_checkpointing
    # training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}
    
    # Finetune the model
    history = ft.finetune(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset['train'],
        lora_config=peft_config, sft_config=training_args
    )
    
    # Save training history
    ft.save_training_history(history, training_args.output_dir)

    # # trainer
    # trainer = SFTTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     #peft_config=peft_config,
    # )

    # # trainer.accelerator.print(f"{trainer.model}")
    # # if hasattr(trainer.model, "print_trainable_parameters"):
    # #     trainer.model.print_trainable_parameters()

    # # train
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)

    # # saving final model
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DatasetArguments, SFTConfig))
    
    # We must assign CUDA_VISIBLE_DEVICES here
    # before transformers and torch are imported
    args, _ = parser.parse_known_args()
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    main(model_args, data_args, training_args)
