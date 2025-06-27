# LLM-loras-train

## train 14B (demo)
```
accelerate launch \
    --config_file ds_config.json \
    --num_processes 1 \
    train_lora.py \
    --model_name_or_path "./models/uncensoredai_UncensoredLM-DeepSeek-R1-Distill-Qwen-14B" \
    --dataset_path "./dataset/uka.json" \
    --output_dir "./lora_finetuned_14B_4bit_uka_uncen" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --num_train_epochs 5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_seq_length 512  \
    --quantization_bits 4
```

## evl 
### base
```
python evaluate_lora.py \
    --eval_mode base \
    --base_model_path "./models/uncensoredai_UncensoredLM-DeepSeek-R1-Distill-Qwen-14B" \
    --eval_dataset_path "./dataset/uka.json" \
    --max_seq_length 512 \
    --per_device_eval_batch_size 4
```
### merged
```
python evaluate_lora.py \
    --eval_mode merged \
    --merged_model_path "./merged_deepseek_14B_uka/" \
    --eval_dataset_path "./dataset/uka.json" \
    --max_seq_length 512 \
    --per_device_eval_batch_size 4
```

### Base Model + LoRA Adapter
```
python evaluate_lora.py \
    --eval_mode base_with_lora \
    --base_model_path "./models/uncensoredai_UncensoredLM-DeepSeek-R1-Distill-Qwen-14B" \
    --lora_adapter_path "./lora_finetuned_14B_4bit_uka_uncen/lora_adapter" \
    --eval_dataset_path "./dataset/uka.json" \
    --max_seq_length 512 \
    --per_device_eval_batch_size 4
```

## merge
```
python merge_lora.py \
    --base_model_path "./models/uncensoredai_UncensoredLM-DeepSeek-R1-Distill-Qwen-14B" \
    --lora_adapter_path "./lora_finetuned_14B_4bit_uka_uncen/lora_adapter" \
    --output_merged_model_path "./merged_deepseek_14B_uka"
```

## chat
```
python chat_bot.py \
    --model_path "./models/uncensoredai_UncensoredLM-DeepSeek-R1-Distill-Qwen-14B" \
    --lora_adapter_path "./lora_finetuned_14B_4bit_uka_uncen/lora_adapter" \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --do_sample
```
