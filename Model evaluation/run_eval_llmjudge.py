import json
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import random
import time
import re
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = "/storage/student6/GalLens_student6"
INPUT_DIR = os.path.join(BASE_DIR, "Metric/model_answers")
OUTPUT_DIR = os.path.join(BASE_DIR, "Metric/final_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 120 
MODEL_NAME = "gemini-2.5-flash"


API_KEYS = []

DISEASE_CLASSES = [
    "avian influenza (head)", "chronic respiratory(head)",
    "healthy (feces)", "salmonella(feces)",
    "bumble foot", "healthy foot", "foot scaly leg mite", "foot spur",
    "fowlpox (head)", "healthy head",
    "new castle (feces)", "new castle disease(head)",
    "Other"
]

BIOBERT_MODEL = "dmis-lab/biobert-v1.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Part I. text metrics calculation
def load_biobert():
    print("📦 Loading BioBERT...")
    tok = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    mod = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
    return tok, mod

def get_bio_embedding(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad(): out = mod(**inputs)
    return out.last_hidden_state.mean(dim=1)

def calc_text_metrics(data, scorer, bio_tok, bio_mod):
    print("   -> Calculating Local Metrics (ROUGE/BioBERT)...")
    for item in tqdm(data, desc="Local Metrics"):
        r = scorer.score(item['ground_truth'], item['model_answer'])
        item['rouge_l'] = r['rougeL'].fmeasure
        emb1 = get_bio_embedding(item['model_answer'], bio_tok, bio_mod)
        emb2 = get_bio_embedding(item['ground_truth'], bio_tok, bio_mod)
        item['biobert_sim'] = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    return data

# Part II. batched Gemini call
def call_gemini_batch(batch_items, retries=3):
    batch_input = []
    for idx, item in enumerate(batch_items):
        batch_input.append({
            "id": idx,
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "model_answer": item['model_answer']
        })
    
    prompt = f"""
    You are a Senior Veterinary Pathologist evaluating AI diagnosis models.
    Process these {len(batch_items)} items.
    
    INPUT DATA: {json.dumps(batch_input, ensure_ascii=False)}

    YOUR TASKS:
    1. **Rate Performance (1-10)**:
       - Accuracy, Relevance, Fluency.
    
    2. **Smart Label Extraction (CRITICAL)**:
       Map 'model_answer' to ONE class from: {json.dumps(DISEASE_CLASSES)}
       
       **RULES FOR BASE MODELS (IMPORTANT):**
       - Base models often describe symptoms without naming the disease. You MUST INFER the label from the description.
       - **Example 1:** Model says "White scales on legs" -> You output "foot scaly leg mite".
       - **Example 2:** Model says "Green watery poop" -> You output "new castle (feces)".
       - **Example 3:** Model says "Swollen head and eyes" -> You output "avian influenza (head)" (or closest match).
       - **Refusals:** If model says "I don't know", "Consult a vet", or gives vague answers like "The bird is sick" (without specific symptoms) -> You output "Other".
       - **Healthy:** If model says "No signs of illness" -> Output "healthy (head/foot/feces)" depending on context.

    OUTPUT JSON LIST:
    [
        {{
            "id": 0,
            "accuracy": 8, "relevance": 9, "fluency": 10,
            "gt_label": "class_name",
            "pred_label": "class_name" 
        }}, ...
    ]
    """

    for i in range(retries):
        try:
            current_key = random.choice(API_KEYS)
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0, "response_mime_type": "application/json"})
            
            response = model.generate_content(prompt)
            try:
                json_str = re.search(r'\[.*\]', response.text, re.DOTALL).group(0)
                results = json.loads(json_str)
                if len(results) != len(batch_items): raise ValueError("Mismatch")
                return results
            except: pass
        except: time.sleep(2 + i)
            
    return [{"id": i, "accuracy": 0, "relevance": 0, "fluency": 0, "gt_label": "Other", "pred_label": "Other"} for i in range(len(batch_items))]

def run_batch_evaluation(data):
    print(f"   -> Running Batch Evaluation (Batch Size: {BATCH_SIZE})...")
    chunks = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    processed_data = []

    for chunk in tqdm(chunks, desc="Processing Batches"):
        results = call_gemini_batch(chunk)
        res_map = {r.get('id'): r for r in results}
        
        for idx, item in enumerate(chunk):
            res = res_map.get(idx, {})
            item.update(res) 
            
            if item.get('gt_label') not in DISEASE_CLASSES: item['gt_label'] = "Other"
            if item.get('pred_label') not in DISEASE_CLASSES: item['pred_label'] = "Other"
            processed_data.append(item)
            
    return processed_data

# main pipeline
def process_file(filename, bio_tok, bio_mod, scorer):
    filepath = os.path.join(INPUT_DIR, filename)
    model_name = filename.replace(".jsonl", "")
    print(f"\nProcessing Model: {model_name}...")
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = calc_text_metrics(data, scorer, bio_tok, bio_mod)
    data = run_batch_evaluation(data)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, f"detailed_{model_name}.csv"), index=False)

    acc = accuracy_score(df['gt_label'], df['pred_label'])
    f1 = f1_score(df['gt_label'], df['pred_label'], average='weighted')
    
    report = {
        "Model": model_name,
        "ROUGE-L": round(df['rouge_l'].mean(), 4),
        "BioBERT Sim": round(df['biobert_sim'].mean(), 4),
        "G-Eval Accuracy": round(df['accuracy'].mean(), 2),
        "G-Eval Relevance": round(df['relevance'].mean(), 2),
        "G-Eval Fluency": round(df['fluency'].mean(), 2),
        "Diagnostic Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4)
    }
    
    cm = confusion_matrix(df['gt_label'], df['pred_label'], labels=DISEASE_CLASSES)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES, cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{model_name}.png"))
    plt.close()
    
    return report

def main():
    bio_tok, bio_mod = load_biobert()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    all_reports = []
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    for f in files:
        rep = process_file(f, bio_tok, bio_mod, scorer)
        all_reports.append(rep)
        with open(os.path.join(OUTPUT_DIR, f"report_{rep['Model']}.json"), 'w') as f_out:
            json.dump(rep, f_out, indent=4)

    final_df = pd.DataFrame(all_reports).set_index("Model")
    final_df.to_csv(os.path.join(OUTPUT_DIR, "FINAL_COMPARISON.csv"))
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(final_df)

if __name__ == "__main__":
    main()