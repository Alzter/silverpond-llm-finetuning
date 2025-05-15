python run_finetune.py \
	`
	# Dataset Configuration
	` \
  --dataset "../data/test/testdata/data_train.csv" \
  --test_size 0.2 \
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
	--use_reentrant False \
  --attn_implementation "sdpa" \
  `
  # Fine-tuning Configuration
  ` \
  --output_dir "../data/test/models/Qwen2.5-FT-Test" \
  --use_peft_lora True \
  `
  # LoraConfig
  ` \
  --lora_r 6 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  `
  # SFTConfig
  # Using the configuration from:
  # https://colab.research.google.com/github/dvgodoy/FineTuningLLMs/blob/main/Chapter0.ipynb
  ` \
  --max_seq_length 64 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --optim "adamw_torch_fused" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant" \
  --packing True\
  --logging_steps 10 \
  --logging_dir './logs' \
  --output_dir FINETUNED_LLM_PATH \
  --report_to 'none' \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 16 \
  --auto_find_batch_size True \
