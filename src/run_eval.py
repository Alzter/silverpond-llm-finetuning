#!/usr/bin/env python
# coding: utf-8

from evaluate import EvaluationConfig
from utils import ModelArguments, EvalDatasetArguments, create_and_prepare_model
from transformers import HfArgumentParser

def main(eval_config : EvaluationConfig, model_args : ModelArguments, data_args : EvalDatasetArguments):
    # Load evaluation dataset
    import preprocess as pre
    eval_dataset, label_names = pre.load_dataset(
        data_args.eval_dataset,
        data_args.text_columns,
        data_args.label_columns
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
    parser = HfArgumentParser((EvaluationConfig, ModelArguments, EvalDatasetArguments))

    eval_config, model_args, data_args = parser.parse_args_into_dataclasses()

    main(eval_config, model_args, data_args)

