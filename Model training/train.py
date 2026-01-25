import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import partial
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

DATA_ROOT = "/home/student6/GalLens_student6/Dataset_images"
TRAIN_JSONL = "train.jsonl"
VAL_JSONL = "val.jsonl"
OUTPUT_DIR = "./checkpoints_qwen"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Hyperparameters
BATCH_SIZE = 1              
GRADIENT_ACCUMULATION = 32  
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.10


class ChickenDiseaseDataset(Dataset):
    def __init__(self, jsonl_path, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.data = []
        
        # Index img files for quick lookup
        self.image_index = {}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.image_index[file] = os.path.join(root, file)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                img_path = item.get('img_path') or item.get('image')
                if img_path:
                    self.data.append(item)
            except: continue
        print(f"Loaded {len(self.data)} items from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            img_path = item.get('img_path') or item.get('image')
            
            # Find image
            full_path = os.path.join(self.root_dir, img_path)
            if not os.path.exists(full_path):
                name = os.path.basename(img_path)
                full_path = self.image_index.get(name, full_path)
            
            # Load img
            image = Image.open(full_path).convert("RGB")
            
            # Resize abit if too large
            if image.width * image.height > 2000*2000:
                image.thumbnail((1024, 1024))

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item['question']},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item['answer']}],
                },
            ]

            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")

            labels = inputs["input_ids"][0].clone()
            # masking
            assistant_tokens = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            if len(assistant_tokens) > 0:
                seq_len = len(assistant_tokens)
                for i in range(len(labels) - seq_len, -1, -1):
                    if torch.all(labels[i:i+seq_len] == torch.tensor(assistant_tokens)):
                        labels[:i+seq_len] = -100
                        break

            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"][0],
                "image_grid_thw": inputs["image_grid_thw"][0],
                "labels": labels,
            }
        except Exception as e:
            print(f"Skip error: {e}")
            return self.__getitem__(random.randint(0, len(self.data)-1))

# Collator standard from Colab (using cat instead of stack)
def data_collator(features, processor=None):
    features = [f for f in features if f is not None]
    if not features: return None

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
    
    # Fix iteration over a 0-d tensor error
    image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0).view(-1, 3) 

    pad_id = processor.tokenizer.pad_token_id if processor else 151643
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
    }

def main():
    torch.manual_seed(42)
    
    print("Loading Processor & Model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1024*28*28)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        device_map="auto"
    )

    # Disable gradient checkpointing
    model.gradient_checkpointing_disable() 
    model.config.use_cache = False # Disable cache to save VRAM during training

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading Data...")
    train_ds = ChickenDiseaseDataset(TRAIN_JSONL, DATA_ROOT, processor)
    val_ds = ChickenDiseaseDataset(VAL_JSONL, DATA_ROOT, processor)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        
        dataloader_num_workers=2,       # Server runs best with 2 workers
        gradient_checkpointing=False,   # Disable gradient checkpointing
        remove_unused_columns=False,    # Required for Qwen
        ddp_find_unused_parameters=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(data_collator, processor=processor)
    )

    print("START TRAINING...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    main()