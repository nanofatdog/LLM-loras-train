accelerate launch \
    --config_file ds_config.json \
    --num_processes 2 \
    train_lora.py \
    --model_name_or_path "./models/thirdeyeai_DeepSeek-R1-Distill-Qwen-7B-uncensored/" \
    --dataset_path "./dataset/uka.json" \
    --output_dir "./lora_finetuned_7B_uka" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16
