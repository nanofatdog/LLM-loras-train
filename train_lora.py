import argparse
import torch
from datasets import load_dataset
# เพิ่ม BitsAndBytesConfig เข้ามาใน import
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # เพิ่ม prepare_model_for_kbit_training
from accelerate import Accelerator
import os

# --- 1. Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning of DeepSeek-R1-Distill-Qwen-1.5B with DeepSpeed")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Path to the pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Name of the dataset from Hugging Face datasets or path to local JSON/CSV file.")
    parser.add_argument("--output_dir", type=str, default="./lora_deepseek_finetuned",
                        help="Where to store the fine-tuned model and logs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                        help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate for AdamW.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.") 
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every X updates steps.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16 (mixed precision) training.")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 (mixed precision) training.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for dataset tokenization.")
    # === เพิ่ม Argument สำหรับ Quantization ===
    parser.add_argument("--quantization_bits", type=int, default=16, 
                        help="Quantization bits for base model (4, 8, 16). 16 means no quantization (BF16/FP16).")

    return parser.parse_args()

# --- ฟังก์ชันสำหรับสร้าง device_map โดยอัตโนมัติ (และแสดง Layer) ---
def generate_auto_device_map(model_path, num_gpus_for_dist, accelerator_obj):
    # โหลดโมเดลในโหมด "meta" เพื่อตรวจสอบโครงสร้างโดยไม่ใช้ VRAM
    # นี่คือวิธีที่ปลอดภัยและรวดเร็วที่สุดในการหา Layers
    # ใช้ low_cpu_mem_usage=True เพื่อลดการใช้ RAM ของ CPU ด้วย
    try:
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, # dtype ใดก็ได้ เพราะแค่ meta
            device_map="auto", # device_map="auto" ในโหมด meta เพื่อให้เข้าถึงโครงสร้างทั้งหมด
            low_cpu_mem_usage=True
        )
    except Exception as e:
        # Fallback สำหรับบางกรณีที่ device_map="auto" อาจมีปัญหา
        accelerator_obj.print(f"Warning: Could not load model in meta device initially with device_map='auto', trying device_map=None. Error: {e}")
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=None, # โหลดลง CPU
            low_cpu_mem_usage=True 
        )
    
    device_map = {}
    layer_names = []
    
    # รวบรวมชื่อโมดูล 'model.layers.X' ทั้งหมด
    for name, module in temp_model.named_modules():
        if "model.layers" in name and isinstance(module, torch.nn.Module):
            try:
                parts = name.split('.')
                # ส่วนสุดท้ายก่อนตัวเลข index คือชื่อ submodule (เช่น self_attn)
                # เราต้องการเพียงชื่อของ parent layer (model.layers.X)
                # เช็คว่าส่วนที่สองนับจากท้ายสุดเป็นตัวเลข เพื่อให้ได้ model.layers.X
                if len(parts) >= 2 and parts[-2].isdigit() and (parts[-1] == "input_layernorm" or parts[-1] == "self_attn"):
                    # เราจะเพิ่มแค่ชื่อของบล็อก Layer หลัก เช่น "model.layers.0"
                    parent_layer_name = ".".join(parts[:-1]) # Reconstruct "model.layers.X"
                    if parent_layer_name not in layer_names:
                        layer_names.append(parent_layer_name)
            except IndexError:
                continue 
    
    # เรียงลำดับชื่อ Layers ตาม Index
    layer_names.sort(key=lambda x: int(x.split('.')[-1])) # Sort by the last part which is the index

    total_actual_layers = len(layer_names)
    accelerator_obj.print(f"Detected {total_actual_layers} actual Transformer layers (e.g., model.layers.0 to model.layers.{total_actual_layers-1}).")
    
    layers_per_gpu = total_actual_layers // num_gpus_for_dist
    remainder = total_actual_layers % num_gpus_for_dist
    
    current_layer_idx = 0
    # กำหนด model.embed_tokens ให้ GPU ตัวแรก
    if hasattr(temp_model, 'model') and hasattr(temp_model.model, 'embed_tokens'):
        device_map["model.embed_tokens"] = 0 

    # กำหนด Layers กระจายไปยัง GPUs
    for i in range(num_gpus_for_dist):
        num_layers_for_this_gpu = layers_per_gpu + (1 if i < remainder else 0)
        for _ in range(num_layers_for_this_gpu):
            if current_layer_idx < total_actual_layers:
                # ใช้ชื่อ Layer เต็มที่รวบรวมมา
                device_map[layer_names[current_layer_idx]] = i
                current_layer_idx += 1
            else:
                break 
    
    # กำหนด model.norm และ lm_head ให้ GPU ตัวสุดท้าย
    last_gpu_id = num_gpus_for_dist - 1
    if hasattr(temp_model, 'model') and hasattr(temp_model.model, 'norm'):
        device_map["model.norm"] = last_gpu_id
    if hasattr(temp_model, 'lm_head'):
        device_map["lm_head"] = last_gpu_id

    # ลบ temporary model เพื่อคืนหน่วยความจำ
    del temp_model
    torch.cuda.empty_cache()

    return device_map

# --- 2. Main Training Function ---
def main():
    args = parse_args()

    # ตรวจสอบจำนวน GPU ที่มี
    total_gpus_available = torch.cuda.device_count()
    if total_gpus_available == 0:
        print("No GPUs detected. Please ensure CUDA is installed and GPUs are available.")
        exit(1)
    else:
        print(f"Total GPUs detected: {total_gpus_available}")
    
    # accelerator จะใช้ num_processes จาก accelerate launch command
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else ("bf16" if args.bf16 else "no")
    )
    
    accelerator.print(f"Using {accelerator.num_processes} GPUs for training (as specified by --num_processes).") 

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right" 

    # === กำหนดค่าสำหรับ Quantization (4-bit, 8-bit) ===
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    quantization_config = None
    if args.quantization_bits == 4:
        accelerator.print("Setting up for 4-bit quantization (QLoRA)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # แนะนำสำหรับ 4-bit
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization_bits == 8:
        accelerator.print("Setting up for 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch_dtype, # compute_dtype ใช้ได้ทั้ง 4-bit และ 8-bit
        )
    elif args.quantization_bits == 16:
        accelerator.print("Training in 16-bit (BF16/FP16) precision, no base model quantization.")
    else:
        accelerator.print("Warning: Invalid --quantization_bits. Only 4, 8, 16 are supported. Defaulting to 16 (no quantization).")
        args.quantization_bits = 16 # Force default to 16

    # === สร้าง device_map โดยอัตโนมัติ ===
    # num_gpus_for_dist มาจาก accelerator.num_processes
    accelerator.print("กำลังสร้าง device_map โดยอัตโนมัติ...")
    device_map = generate_auto_device_map(args.model_name_or_path, accelerator.num_processes, accelerator)
    accelerator.print(f"สร้าง device_map สำเร็จ: {device_map}")
    
    # โหลดโมเดลจริงด้วย quantization_config และ device_map ที่สร้างขึ้น
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config, # ส่ง quantization_config ที่สร้างขึ้น
        device_map=device_map, # ใช้ device_map ที่สร้างโดยอัตโนมัติ
    )
    
    # สำคัญ: ต้องเรียก enable_input_require_grads และ prepare_model_for_kbit_training (ถ้าใช้ quantization)
    model.enable_input_require_grads() 
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    actual_target_modules = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name:
                actual_target_modules.append(target)
    actual_target_modules = list(set(actual_target_modules)) 

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=actual_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    model.gradient_checkpointing_enable() 

    # Load and Preprocess Dataset
    try:
        if args.dataset_path.endswith(".json") or args.dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=args.dataset_path)
        else:
            raise ValueError("Dataset path must be a path to a .json or .jsonl file.")
    except Exception as e:
        accelerator.print(f"Error loading dataset from {args.dataset_path}: {e}")
        accelerator.print("Please ensure the dataset file exists and is correctly formatted.")
        exit(1)
    
    if 'train' not in dataset:
        raise ValueError("Dataset must have a 'train' split.")
    
    def format_alpaca_example(examples): 
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            output_text = examples["output"][i]

            if input_text:
                full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            texts.append(full_text)
        return {"text": texts} 

    formatted_dataset = dataset["train"].map(
        format_alpaca_example,
        batched=True, 
        remove_columns=dataset["train"].column_names, 
        num_proc=args.num_workers,
        load_from_cache_file=True,
        desc="Formatting dataset for Causal LM",
    )

    def tokenize_function(examples):
        output = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            return_overflowing_tokens=True, 
            return_length=True,
        )
        return output

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"], 
        num_proc=args.num_workers,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="adamw_torch", 
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="tensorboard", 
        push_to_hub=False, 
        ddp_find_unused_parameters=False, 
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    accelerator.print("Starting training...")
    trainer.train()

    accelerator.print(f"Saving LoRA adapter to {args.output_dir}/lora_adapter")
    trainer.model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")

    accelerator.print("Training complete and adapter saved!")

if __name__ == "__main__":
    main()
