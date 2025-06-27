import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora_adapter(base_model_name_or_path, lora_adapter_path, output_merged_model_path):
    print(f"Loading base model from: {base_model_name_or_path}")
    # ตรวจสอบว่ามี GPU หรือไม่ ถ้ามี ให้ใช้ GPU ถ้าไม่มี ให้ใช้ CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # กำหนด torch_dtype ให้ตรงกับการฝึกของคุณ (bf16 ถ้าใช้) หรือ float16/float32
    # หาก GPU ไม่รองรับ bf16 ก็ใช้ float16 หรือ float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. โหลด Base Model
    # ใช้ device_map="auto" ถ้ามีหลาย GPU หรือ device_map=device ถ้าใช้ GPU เดียว
    # หากโหลดลง GPU เดียวก็ระบุ device=device
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device # โหลดลง GPU หรือ CPU
    )
    print("Base model loaded.")

    # โหลด Tokenizer ของ Base Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    print("Tokenizer loaded.")

    # 2. โหลด LoRA Adapter
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("LoRA adapter loaded.")

    # 3. Merge Model
    print("Merging LoRA weights into the base model...")
    merged_model = model_with_lora.merge_and_unload()
    print("Merging complete.")

    # 4. บันทึกโมเดลที่ Merge แล้ว
    print(f"Saving merged model to: {output_merged_model_path}")
    os.makedirs(output_merged_model_path, exist_ok=True)
    merged_model.save_pretrained(output_merged_model_path)
    tokenizer.save_pretrained(output_merged_model_path)
    print("Merged model and tokenizer saved successfully!")

if __name__ == "__main__":
    # === คุณต้องแก้ไข Path เหล่านี้ตามการตั้งค่าของคุณ ===
    # Path ไปยัง Base Model ที่คุณใช้ Fine-tune
    # ตัวอย่าง: ถ้าโมเดลอยู่ใน ../models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B
    base_model_path = "../models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B" 
    
    # Path ไปยัง LoRA Adapter ที่คุณ Fine-tune ได้มา (Output จาก train_lora.py)
    # ตัวอย่าง: ถ้า output_dir ของคุณคือ ./lora_finetuned_deepseek_model_alpaca
    # LoRA adapter จะถูกบันทึกที่นั่นภายใต้ lora_adapter
    lora_adapter_location = "./lora_finetuned_deepseek_model_alpaca/lora_adapter"

    # Path ที่คุณต้องการบันทึกโมเดลที่ Merge แล้ว
    output_directory_for_merged_model = "./merged_deepseek_lora_model" 
    # ====================================================

    merge_lora_adapter(
        base_model_name_or_path=base_model_path,
        lora_adapter_path=lora_adapter_location,
        output_merged_model_path=output_directory_for_merged_model
    )
