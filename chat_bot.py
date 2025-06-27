import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # เพิ่ม import PeftModel
import argparse

def create_prompt(instruction, user_input=""):
    """
    สร้าง Prompt ตามรูปแบบ Alpaca/UKA instruction format
    """
    if user_input:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

def main():
    parser = argparse.ArgumentParser(description="Simple LLM Chatbot in Terminal.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the directory of the base model or the merged model to use.")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Optional: Path to the directory of the LoRA adapter. If provided, model_path will be treated as base model.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate in response.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation. Higher means more randomness.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter for generation.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling (True) or greedy decoding (False).")
    
    args = parser.parse_args()

    print(f"กำลังโหลดโมเดล Chatbot...")
    
    # กำหนด dtype สำหรับการโหลดโมเดล: bf16 ถ้า GPU รองรับ, ไม่เช่นนั้นใช้ float16/float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    # === ส่วนการโหลดโมเดล (Base + LoRA หรือ Merged Model) ===
    if args.lora_adapter_path:
        print(f"  - โหลด Base Model จาก: {args.model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, # model_path คือ base_model_path
            torch_dtype=torch_dtype,
            device_map="auto" 
        )
        print(f"  - โหลด LoRA adapter จาก: {args.lora_adapter_path}")
        # Apply LoRA adapter กับ base model
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
        print("  - โหลด Base Model และ LoRA adapter สำเร็จแล้ว (สำหรับ Inference Dynamic).")
    else:
        print(f"  - โหลด Merged Model หรือ Base Model โดยตรงจาก: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" 
        )
        print("  - โหลดโมเดลสำเร็จแล้ว (สำหรับ Inference โดยตรง).")

    # ตั้งค่าโมเดลให้อยู่ในโหมด evaluation (สำหรับ Inference)
    model.eval() 
    print("ตั้งค่าโมเดลเป็นโหมดประเมินผล.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left" 

    print("\n======================================")
    print("ยินดีต้อนรับสู่ Chatbot! พิมพ์ 'exit' เพื่อออก")
    print("======================================")

    while True:
        user_instruction = input("\nคุณ: ")
        if user_instruction.lower() == 'exit':
            break
        
        user_input_for_model = "" 

        prompt = create_prompt(user_instruction, user_input_for_model)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("บอท: กำลังคิด...")
        with torch.no_grad(): 
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.eos_token_id, 
            )
        
        response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        generated_response = response_text[len(prompt):].strip() 
        
        print(f"บอท: {generated_response}")

if __name__ == "__main__":
    main()
