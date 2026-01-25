#to make sure all the qna pairs can detect the disease correctly
import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from itertools import cycle

API_KEYS = []

MODEL_NAME = 'gemini-2.5-flash' 
BATCH_SIZE = 120 

BASE_DATA_DIR = r"D:\Chicken-Farm-System\VQA\Dataset Final"
OUTPUT_DIR = r"D:\Chicken-Farm-System\VQA\Data_ref_Final_46k"

FILE_MAPPING = {
    "scaly_class1.jsonl":             "Scaly Leg Mite",
    "Avian_Influenza_2.jsonl":        "Avian Influenza",
    "head_CRD_QA.jsonl":              "Chronic Respiratory Disease",
    "ncd_head_class12.jsonl":         "Newcastle Disease (Head)",
    "fowplox_class1.jsonl":           "Fowl Pox",
    "bumble_class12.jsonl":           "Bumblefoot",
    "spur_class1.jsonl":             "Spurs",
    "healthy_foot_class12.jsonl":     "Healthy Foot",
    "healthy_head_class12.jsonl":     "Healthy Head",
    "healthy_disease_class12.jsonl":  "Healthy (Feces)",
    "Salmonella_class1+2.jsonl":      "Salmonella",
    "ncd_disease_class12.jsonl":      "Newcastle Disease (Feces)"
}

key_iterator = cycle(API_KEYS)
current_key = next(key_iterator)
model = None

def initialize_model():
    global model, current_key
    genai.configure(api_key=current_key)
    model = genai.GenerativeModel(MODEL_NAME)

def switch_key():
    global current_key
    try:
        current_key = next(key_iterator)
        print(f"\nRate Limit/Error -> change key: ...{current_key[-6:]}")
        initialize_model()
        time.sleep(1) 
    except Exception as e:
        print(f"Critical Error switching keys: {e}")

initialize_model()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# backend function to process each batch with retry and key rotation
def process_batch_with_rotation(batch_data, target_disease, max_retries=15):
    input_json_str = json.dumps(batch_data, ensure_ascii=False, indent=1)
    prompt = f"""
    Role: Veterinary Data Editor.
    Task: Process the list of Q&A pairs below. For EACH item, refine the 'answer' strictly following the logic below.

    TARGET DISEASE: "{target_disease}"

    LOGIC FLOW & RULES (Apply to each item independently):

    1. PHASE 1: RELEVANCE CHECK (The "Background" Filter)
       - Analyze the 'q' (Question).
       - IF the question asks about: Image quality, Lighting, Background, Camera angle, Framing, Blur, or Generic Color (non-symptomatic)...
       - ...THEN: Keep 'a' (Answer) EXACTLY AS IS. Do NOT add the disease name.

    2. PHASE 2: MEDICAL INTEGRATION (For Symptom/Health Questions)
       - IF the question relates to: Symptoms, Body parts, Health status, Lesions, Diagnosis...
       - ...THEN: Integrate "{target_disease}" into the answer naturally if it's missing.

    3. PHASE 3: EDITING RULES (For Phase 2 only)**
       - Visual Preservation: You MUST keep the original visual adjectives (e.g., "swollen", "yellow crusts", "purple") word-for-word. These match the pixels.
       - Smooth Connection: Use natural transitions based on context.
       - Differential Logic: If the question asks about a WRONG disease (Negative Sample), explain the visual mismatch and conclude with "{target_disease}".
       
    CRITICAL RULE: LENGTH PRESERVATION
    - Do NOT summarize or shorten the answer. - The refined answer must retain ALL details from the original answer.
    - Adding the disease name should INCREASE the length slightly, never decrease it.

    INPUT DATA (JSON List):
    {input_json_str}

    OUTPUT REQUIREMENT:
    - Return a VALID JSON LIST.
    - Format: [ {{"id": <original_id>, "refined_answer": "<new_string>"}}, ... ]
    - STRICTLY maintain the "id" mapping.
    """

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "resource exhausted" in error_msg or "overloaded" in error_msg or "internal" in error_msg:
                switch_key() 
            else:
                print(f"Error (Attempt {attempt+1}): {e} -> Retrying...")
                time.sleep(2)
    return None 

def find_file_recursive(base_dir, filename):
    for root, dirs, files in os.walk(base_dir):
        if filename in files: return os.path.join(root, filename)
    return None

print(f"Start with {len(API_KEYS)} KEYS and BATCH {BATCH_SIZE}")

for filename, disease_name in FILE_MAPPING.items():
    input_path = find_file_recursive(BASE_DATA_DIR, filename)
    
    if not input_path:
        print(f"File not found: {filename} -> Skipping (recheck the filename)")
        continue
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    print(f"\nProcessing: {filename} ({disease_name})")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    chunks = [all_lines[i:i + BATCH_SIZE] for i in range(0, len(all_lines), BATCH_SIZE)]
    refined_results = []
    
    pbar = tqdm(chunks, desc="Processing Batches")
    for chunk in pbar:
        batch_input = []
        original_map = {}
        
        for idx, line in enumerate(chunk):
            try:
                data = json.loads(line)
                item = {"id": idx, "q": data.get("question", ""), "a": data.get("answer", "")}
                batch_input.append(item)
                original_map[idx] = data 
            except: continue
        
        if not batch_input: continue

        api_results = process_batch_with_rotation(batch_input, disease_name)
        
        if api_results and isinstance(api_results, list):
            result_map = {item['id']: item['refined_answer'] for item in api_results if 'id' in item and 'refined_answer' in item}
            for idx in range(len(chunk)):
                original_data = original_map.get(idx)
                if not original_data: continue
                if idx in result_map:
                    original_data['answer'] = result_map[idx]
                if 'img_path' in original_data: 
                    original_data['image_path'] = original_data.pop('img_path')
                refined_results.append(json.dumps(original_data))
        else:
            for idx in range(len(chunk)):
                original_data = original_map.get(idx)
                if original_data:
                    refined_results.append(json.dumps(original_data))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in refined_results:
                f.write(line + '\n')

    print(f"Completed file {filename}")
print("All files processed")