#!/usr/bin/env python
# coding: utf-8

import os
import sys

from evaluate import EvaluationConfig
from utils import ModelArguments, DatasetArguments, create_and_prepare_model
from transformers import HfArgumentParser

def main(eval_config : EvaluationConfig, model_args : ModelArguments, data_args : DatasetArguments):
    # Load evaluation dataset
    import preprocess as pre
    eval_dataset, label_names = pre.load_dataset(
        data_args.dataset,
        data_args.text_columns,
        data_args.label_columns,
        test_size=0,
        ratio=data_args.ratio
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
    parser = HfArgumentParser((EvaluationConfig, ModelArguments, DatasetArguments))
    
    # If we pass only one argument to the script and it's
    # the path to a json file, let's parse it to get our arguments.

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        # We must assign CUDA_VISIBLE_DEVICES here
        # before transformers and torch are imported
        file = os.path.abspath(sys.argv[1])
        import json
        with open(file) as f:
            data = json.load(f)
            f.close()
        if data.get("cuda_devices"):
            os.environ["CUDA_VISIBLE_DEVICES"] = data["cuda_devices"]
        del data

        args = parser.parse_json_file(json_file=file)
    
    else:
        # We must assign CUDA_VISIBLE_DEVICES here
        # before transformers and torch are imported
        args, _ = parser.parse_known_args()
        if args.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

        args = parser.parse_args_into_dataclasses()

    eval_config, model_args, data_args = args

    main(eval_config, model_args, data_args)

