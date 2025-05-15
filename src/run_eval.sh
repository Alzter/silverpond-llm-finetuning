python run_eval.py \
	`
	# Evaluation Configuration
	` \
	--technique_name "Zero-shot" \
	--max_tokens 10 \
	--prompt None \
	--prompt_role "system" \
	--out_path "../data/test"\
	`
	# Dataset Configuration
	` \
	--eval_dataset "../data/test/data_eval.csv" \
	--text_columns "content" \
	--label_columns "label" \
	`
	# LLM Configuration
	` \
	--model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
	--cuda_devices "1" \
	--use_4bit_quantization True \
	--bnb_4bit_quant_type "nf4" \
	--bnb_4bit_compute_dtype "float16" \
	--use_nested_quant True \
	--use_reentrant True
