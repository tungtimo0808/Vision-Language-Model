# 🐔 GalLens: Vision–Language System for Poultry Disease Diagnosis and Explanation

GalLens is a Vision–Language based system for **chicken disease classification and explanation**, designed to support **non-expert users** in poultry farming.  
The system combines **deep learning, vision–language models, and retrieval-augmented generation (RAG)** to provide both **accurate diagnosis** and **reliable, easy-to-understand explanations**.

This repository accompanies the undergraduate thesis:

> **"Vision–Language Based Poultry Disease Diagnosis and Explanation System"**  
> Faculty of Information Technology  
> Foreign Trade University (FTU)

---

## 🎯 Objectives

- Automatically **classify poultry diseases** from images
- Generate **natural language explanations** for the diagnosis
- Reduce **hallucination** by grounding answers in verified medical knowledge
- Support **non-expert users** (farmers, students, technicians)

---

## 🧠 System Overview

The system consists of **three main phases**:

### 1️⃣ Dataset Construction

- Build a **domain-specific VQA dataset** for poultry diseases
- Use **AI-assisted generation (Gemini 2.5 Flash)** + **human verification**
- Ensure:
  - Visual grounding
  - Medical correctness
  - Consistent terminology

### 2️⃣ Model Fine-tuning

- Base model: **Qwen2-VL-7B Instruct**
- Fine-tuning method: **LoRA**
- Multiple configurations tested:
  - Only Attention layers
  - Full Linear layers (Final model: **GalLens-Expert**)
- Results show:
  - Base model has **almost no domain knowledge**
  - Fine-tuned models achieve **large improvements**
  - GalLens-Expert performs best in both **classification and explanation quality**

### 3️⃣ Retrieval-Augmented Generation (RAG)

- Build a **medical knowledge base** from trusted sources (PDF, text, guidelines)
- Use an **embedding model + vector database**
- At inference time:
  - Retrieve relevant medical knowledge
  - Inject it into the VLM prompt
- Benefits:
  - Reduce hallucination
  - Improve factual correctness for **treatment-related answers**

---

## 📊 Experimental Results

- **Confusion matrix analysis** shows:
  - Base model collapses most predictions into “Other”
  - Fine-tuned model can distinguish visually similar diseases
- **Quantitative metrics**:
  - ROUGE-L, BERT Similarity
  - G-Eval (Accuracy, Relevance, Fluency)
  - Classification Accuracy, F1-score, Recall
- Final model (**GalLens-Expert**) achieves **best performance on all metrics**

---

## 🏗️ Project Structure (Suggested)

