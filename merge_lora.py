import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse # เพิ่ม import argparse

def merge_lora_adapter(base_model_name_or_path, lora_adapter_path, output_merged_model_path):
    print(f"กำลังโหลด Base Model จาก: {base_model_name_or_path}")
    
    # กำหนด dtype สำหรับการโหลดโมเดล: bf16 ถ้า GPU รองรับ, ไม่เช่นนั้นใช้ float16/float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. โหลด Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto", # ใช้ device_map="auto" เพื่อโหลดลง GPU ที่เหมาะสม (หรือ CPU ถ้าไม่มี GPU)
    )
    print("โหลด Base Model สำเร็จแล้ว.")

    # โหลด Tokenizer ของ Base Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    print("โหลด Tokenizer สำเร็จแล้ว.")

    # 2. โหลด LoRA Adapter
    print(f"กำลังโหลด LoRA adapter จาก: {lora_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("โหลด LoRA adapter สำเร็จแล้ว.")

    # 3. Merge Model
    print("กำลังรวม LoRA weights เข้ากับ Base Model...")
    merged_model = model_with_lora.merge_and_unload()
    print("รวมโมเดลเสร็จสมบูรณ์.")

    # 4. บันทึกโมเดลที่ Merge แล้ว
    print(f"กำลังบันทึกโมเดลที่รวมแล้วไปยัง: {output_merged_model_path}")
    os.makedirs(output_merged_model_path, exist_ok=True)
    merged_model.save_pretrained(output_merged_model_path)
    tokenizer.save_pretrained(output_merged_model_path)
    print("บันทึกโมเดลและ Tokenizer ที่รวมแล้วสำเร็จ!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into a base model.")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the directory of the base model.")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to the directory of the LoRA adapter (e.g., ./output_dir/lora_adapter).")
    parser.add_argument("--output_merged_model_path", type=str, required=True,
                        help="Path to the directory where the merged model will be saved.")
    
    args = parser.parse_args()

    merge_lora_adapter(
        base_model_name_or_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_merged_model_path=args.output_merged_model_path
    )
