import json
import torch
import os
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


BASE_DIR = "/storage/student6/GalLens_student6"
TEST_FILE = os.path.join(BASE_DIR, "Test/Test_Final.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "Metric/model_answers")
IMAGE_ROOT = BASE_DIR 

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
CACHE_DIR = "/storage/student6/mustela_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODELS_TO_RUN = {
    "base_model": None,
    "expert_model": os.path.join(BASE_DIR, "final_lora_a100"),
    "full_attention_only": os.path.join(BASE_DIR, "Benchmark/Full_AttentionOnly"),
    "nano_10_percent": os.path.join(BASE_DIR, "Benchmark/Nano_10Percent"),
    "nano_40_percent": os.path.join(BASE_DIR, "Benchmark/Nano_40Percent"),
}

def load_base_model():
    print(f"📦 Loading Base Model: {MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", # Eager -> flash attention 2 would be better if installed
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, min_pixels=256*28*28, max_pixels=1280*28*28)
    return model, processor

def find_image_path(rel_path):
    candidates = [
        os.path.join(IMAGE_ROOT, rel_path),
        os.path.join(IMAGE_ROOT, "Dataset_images", rel_path),
        os.path.join(IMAGE_ROOT, rel_path.replace("Dataset_images/", ""))
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def run_inference(model_name, model, processor, data, is_lora=False):
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}.jsonl")
    print(f"\nRunning Inference for: {model_name}")
    print(f"Saving to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc=f"Generating {model_name}"):
            # img processing
            rel_path = item.get('img_path') or item.get('image')
            if not rel_path: continue
            
            image_path = find_image_path(rel_path)
            if not image_path:
                # Log error if image not found (for debugging)
                res = item.copy()
                res['model_answer'] = "ERROR: Image not found"
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                continue

            # 2. prompt
            question = item.get('question', "Describe this image.")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]}
            ]
            
            # 3. inference
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                image_obj = Image.open(image_path).convert("RGB")
                inputs = processor(
                    text=[text_prompt], images=[image_obj], padding=True, return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                
                generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            except Exception as e:
                prediction = f"ERROR: {str(e)}"

            # 4. save result
            result_item = {
                "img_path": rel_path,
                "question": question,
                "ground_truth": item.get('answer', ""),
                "model_answer": prediction,
                "model_version": model_name
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')


def main():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return

    print(f"Reading Test Data from {TEST_FILE}...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Found {len(test_data)} samples.")

    base_model, processor = load_base_model()

    for name, lora_path in MODELS_TO_RUN.items():
        print(f"\n" + "-"*40)
        
        current_model = base_model
        
        # If the model has LoRA -> Attach Adapter
        if lora_path:
            if os.path.exists(lora_path):
                print(f"Attaching LoRA: {lora_path}")
                # Load adapter
                current_model = PeftModel.from_pretrained(base_model, lora_path)
                current_model.eval()
                
                # Run inference
                run_inference(name, current_model, processor, test_data, is_lora=True)
                
                # Unload adapter to return to clean base model for next loop
                print("Unloading LoRA...")
                current_model.unload() # Return to original base_model
                current_model = base_model 
            else:
                print(f"LoRA path not found: {lora_path}. Skipping {name}.")
                continue
        else:
            # If Base model -> Run directly
            print("Running Base Model (No Adapter)")
            run_inference(name, base_model, processor, test_data, is_lora=False)

    print("\n" + "="*50)
    print('done')

if __name__ == "__main__":
    main()