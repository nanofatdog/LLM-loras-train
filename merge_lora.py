import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse 

def merge_lora_adapter(base_model_name_or_path, lora_adapter_paths, output_merged_model_path, scaling_factor=1.0):
    print(f"กำลังโหลด Base Model จาก: {base_model_name_or_path}")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. โหลด Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto", 
    )
    print("โหลด Base Model สำเร็จแล้ว.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    print("โหลด Tokenizer สำเร็จแล้ว.")

    # 2. โหลด LoRA Adapter(s)
    model_with_lora = None
    if not lora_adapter_paths:
        raise ValueError("ต้องระบุอย่างน้อย 1 LoRA adapter path")

    # โหลด LoRA adapter ตัวแรก
    first_adapter_path = lora_adapter_paths[0]
    print(f"กำลังโหลด LoRA adapter แรก จาก: {first_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(base_model, first_adapter_path)
    print("โหลด LoRA adapter แรกสำเร็จ.")

    # หากมี LoRA adapter ตัวที่สองเป็นต้นไป ให้โหลดเพิ่ม
    if len(lora_adapter_paths) > 1:
        print(f"กำลังโหลด LoRA adapter เพิ่มเติม ({len(lora_adapter_paths) - 1} ตัว)...")
        for i, adapter_path in enumerate(lora_adapter_paths[1:]):
            adapter_name = f"adapter_{i+1}" # สร้างชื่อ adapter เช่น adapter_1, adapter_2
            print(f"  - กำลังโหลด adapter '{adapter_name}' จาก: {adapter_path}")
            model_with_lora.load_adapter(adapter_path, adapter_name=adapter_name)
            # PEFT จะทำให้ adapter ที่โหลดล่าสุดเป็น active โดยอัตโนมัติ
            # merge_and_unload จะรวมทุก adapter ที่ active
            print(f"  - โหลด adapter '{adapter_name}' สำเร็จ.")
        print("โหลด LoRA adapter เพิ่มเติมทั้งหมดสำเร็จแล้ว.")
        # หากมีหลาย adapter, สามารถตั้งค่าให้ทั้งหมด active พร้อมกันเพื่อ merge (default PEFT behavior)
        # หรือเลือกเฉพาะบางตัว ถ้าใช้ add_weighted_adapter
        # สำหรับการ merge แบบง่ายๆ คือโหลดทั้งหมด และ merge_and_unload จะรวมทุก active adapter
        
    print("LoRA adapter(s) พร้อมสำหรับการรวม.")

    # 3. Merge Model
    print(f"กำลังรวม LoRA weights เข้ากับ Base Model ด้วย scaling_factor: {scaling_factor}...")
    # scaling_factor จะถูกนำไปใช้กับ LoRA adapter ที่ Active ทั้งหมด
    merged_model = model_with_lora.merge_and_unload(scaling_factor=scaling_factor)
    print("รวมโมเดลเสร็จสมบูรณ์.")

    # 4. บันทึกโมเดลที่ Merge แล้ว
    print(f"กำลังบันทึกโมเดลที่รวมแล้วไปยัง: {output_merged_model_path}")
    os.makedirs(output_merged_model_path, exist_ok=True)
    merged_model.save_pretrained(output_merged_model_path)
    tokenizer.save_pretrained(output_merged_model_path)
    print("บันทึกโมเดลและ Tokenizer ที่รวมแล้วสำเร็จ!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge one or more LoRA adapters into a base model with a global scaling factor.")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the directory of the original base model.")
    # ใช้ nargs='+' เพื่อรับ Path ได้หลายตัว (คั่นด้วย spacebar)
    parser.add_argument("--lora_adapter_paths", type=str, nargs='+', required=True,
                        help="List of paths to the directories of LoRA adapters to merge. Separate paths by space (e.g., './adapter1' './adapter2').")
    parser.add_argument("--output_merged_model_path", type=str, required=True,
                        help="Path to the directory where the merged model will be saved.")
    parser.add_argument("--scaling_factor", type=float, default=1.0,
                        help="Global scaling factor to apply to all LoRA adapters during merging. Default is 1.0.")
    
    args = parser.parse_args()

    merge_lora_adapter(
        base_model_name_or_path=args.base_model_path,
        lora_adapter_paths=args.lora_adapter_paths,
        output_merged_model_path=args.output_merged_model_path,
        scaling_factor=args.scaling_factor
    )
