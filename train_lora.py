import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling 
from peft import LoraConfig, get_peft_model 
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
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, # batch size per GPU
                        help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, # กลับมาใช้ค่ากลางๆ
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate for AdamW.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.") # กลับมาใช้ 512
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every X updates steps.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16 (mixed precision) training.")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 (mixed precision) training.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for dataset tokenization.")

    return parser.parse_args()

# --- 2. Main Training Function ---
def main():
    args = parse_args()

    # Initialize Accelerator (handles DeepSpeed integration)
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else ("bf16" if args.bf16 else "no")
    )
    
    accelerator.print(f"Using {accelerator.num_processes} GPUs for training.") 

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right" # Important for Causal LM

    # Load Model
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    
    model.enable_input_require_grads() 

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

    # เปิดใช้งาน gradient checkpointing เพื่อประสิทธิภาพ VRAM (แนะนำอย่างยิ่งสำหรับ LLMs)
    # === แก้ไขตรงนี้: ลบ use_reentrant=False ออก ===
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

    # Training Arguments
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

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    accelerator.print("Starting training...")
    trainer.train()

    # Save the fine-tuned LoRA adapter
    accelerator.print(f"Saving LoRA adapter to {args.output_dir}/lora_adapter")
    trainer.model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")

    accelerator.print("Training complete and adapter saved!")

if __name__ == "__main__":
    main()
