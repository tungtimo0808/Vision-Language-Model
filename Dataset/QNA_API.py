import os
import json
import time
import sys
import google.generativeai as genai
from PIL import Image

API_KEY = ""
IMAGE_FOLDER_PATH = r"dataset_chicken/leg/h_leg_1" 
OUTPUT_FILE = "healthy_foot_1.jsonl" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash') 

SYSTEM_PROMPT = """
You are an expert AI Veterinarian. Your task is to generate a high-quality VQA (Visual Question Answering) dataset for poultry leg diseases.

Target Classes (Strictly 4):
1. Healthy
2. Scaly Leg Mites
3. Bumblefoot
4. Spur Foot

CORE RULES (MUST FOLLOW):
1. Visual Grounding ONLY: Do NOT answer based on general medical knowledge. Answer ONLY based on what is visible in the pixels.
2. Three-Pillar Logic: Every response must address one of these aspects:
    -   Part: Where is it? (Leg, Shank, Toes, Claw).
    -   Identity: What is it? (Diagnosis).
    -   Evidence: Why? (Pixel-level justification).

Generate 8-10 QnA pairs following this strict structure:

Group 1: Anatomy & Localization (Criterion 3 - Body Part)
-   Focus: Identify the specific part shown (Shank, toes, footpad, hock).
-   *Constraint:* Force the model to look. E.g., "Which part of the chicken is visible?" -> "The image shows the **lower shank and toes**."

Group 2: Diagnosis & Identification (Criterion 1 - Identity)
-   Focus: The direct classification of the disease/status.
-   *Style:* Mix between direct questions ("What disease is this?") and user-style queries ("Is my chicken sick?").
-   *Answer:* Must state the class name clearly (e.g., "Healthy", "Scaly Leg Mites").

Group 3: Visual Evidence & Reasoning (Criterion 2 - Evidence)
-   The "BECAUSE" Logic: This is the most important part.
-   Task: Connect a specific visual feature to the diagnosis.
-   Template for Healthy: "Why is this foot considered Healthy?" -> "Because the scales are smooth and lie flat (ruling out Mites) and the footpad has no black scabs (ruling out Bumblefoot)."
-   Template for Disease: "What visual feature confirms [Disease X]?" -> "The presence of [specific feature like 'lifted crusty scales' or 'black ulcer'] visible on the [specific location] confirms the diagnosis."
-   Visual Details: Ask about color (yellow/red/black), texture (smooth/rough/crusty), shape (swollen/normal).

OUTPUT FORMAT:
-   Return ONLY a valid JSON list of objects.
-   Do NOT use Markdown.
-   Format: [{"question": "...", "answer": "..."}, ...]
"""

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def get_processed_images(jsonl_path):
    """Đọc file jsonl để xem ảnh nào đã làm rồi (Resume capability)"""
    processed = set()
    if not os.path.exists(jsonl_path): return processed
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(record['img_path'])
            except: continue
    return processed

def generate_qna(image_path):
    try:
        img = Image.open(image_path)
        response = model.generate_content([SYSTEM_PROMPT, img])
        return json.loads(clean_json_string(response.text))
    except Exception as e:
        print(f"\n Error creating qna for {os.path.basename(image_path)}: {e}")
        return None


def main():
    # 1. check img processed
    processed_images = get_processed_images(OUTPUT_FILE)
    print(f"🔄 Found {len(processed_images)} old images in file (will skip).")

    # 2. load images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.lower().endswith(valid_extensions)]
    
    print(f"{len(image_files)} imgs. Starting processing")
    
    count_session = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for idx, filename in enumerate(image_files):
            full_path = os.path.join(IMAGE_FOLDER_PATH, filename)
            norm_path = full_path.replace("\\", "/") 
            
            # img already processed -> skip
            if norm_path in processed_images: continue
                
            print(f"⚡ [{idx+1}/{len(image_files)}] Flash processing: {filename}...", end="", flush=True)
            
            qna_list = generate_qna(full_path)
            
            if qna_list:
                for item in qna_list:
                    record = {"img_path": norm_path, "question": item['question'], "answer": item['answer']}
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    processed_images.add(norm_path)
                
                f_out.flush() 
                print(f" Done ({len(qna_list)} QnA pairs)")
                count_session += 1
            else:
                print(" ⚠️ Skipped (API Error).")

            # wait between requests
            wait_time = 4
            for i in range(wait_time, 0, -1):
                sys.stdout.write(f"\r⏳ Waiting {i}s... ")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\r" + " "*20 + "\r")

            if count_session >= 1450:
                print("\nApproaching Flash's daily limit")
                break

    print(f"\n Session complete! Data saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()