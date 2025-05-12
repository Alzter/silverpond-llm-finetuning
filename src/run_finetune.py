import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer
from utils import create_and_prepare_model

from utils import ModelArguments, DatasetArguments

def main(model_args, data_args, training_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = model_args.cuda_devices
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}
    
    # Load training dataset
    import preprocess as pre
    train_dataset, label_names = pre.load_dataset(
        data_args.dataset_name_or_path,
        data_args.text_columns,
        data_args.label_columns,
        test_size=0
    ) 

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
    
    # trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
