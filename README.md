# 🐔 GalLens: Vision–Language System for Poultry Disease Diagnosis and Explanation

GalLens is a Vision–Language based system for **chicken disease classification and explanation**, designed to support **non-expert users** in poultry farming.  
The system combines **deep learning, vision–language models (VLMs), and Retrieval-Augmented Generation (RAG)** to provide both **accurate diagnosis** and **reliable, knowledge-grounded explanations**.

This repository accompanies the undergraduate thesis:

> **Vision–Language Based Poultry Disease Diagnosis and Explanation System**  
> Author: Nguyễn Hoàng Tùng  
> USTH

---

# 📌 Table of Contents

- Overview
- System Architecture
- Full Workflow
- Dataset Construction
- Model Fine-tuning
- Retrieval-Augmented Generation (RAG)
- Inference Pipeline
- Experimental Results
- Project Structure
- Technologies
- Limitations & Future Work

---

# 🧠 Overview

GalLens aims to solve two problems at the same time:

1. **What disease does this chicken have?** (classification)
2. **Why and what should I do?** (explanation + treatment)

Unlike normal CNN classifiers, GalLens is a **Vision–Language system** that:
- Understands images
- Understands natural language questions
- Generates medical explanations grounded in real documents

---

# 🏗️ System Architecture

<!-- INSERT FIGURE: Full system overview (3-phase pipeline: Dataset → Fine-tuning → RAG) -->
> 📌 Use: Figure "Overview of the dataset construction process" / Phase diagram in your thesis

The system consists of **three main phases**:

1. Dataset Construction
2. Vision–Language Model Fine-tuning
3. Knowledge-grounded Inference using RAG

---

# 🔄 Full Workflow

<!-- INSERT FIGURE: End-to-end workflow from user input to final answer -->
> 📌 Use: Phase 3 RAG workflow diagram in your thesis

High-level workflow:

1. User inputs: **Image + Question**
2. System routes the query:
   - Diagnosis → VLM
   - Treatment / Definition → RAG + VLM
3. If RAG is needed:
   - Retrieve documents
   - Inject knowledge into prompt
4. VLM generates:
   - Disease label
   - Natural language explanation

---

# 📦 Dataset Construction

<!-- INSERT FIGURE: Phase 1 Data Preparation pipeline -->
> 📌 Use: "Overview of Phase 1: Data Preparation"

Steps:

1. Collect raw poultry disease images
2. Use **Gemini 2.5 Flash** to generate draft VQA pairs
3. Store results in JSONL format
4. Perform **human verification**
5. Reject or fix mislabeled samples
6. Build **cleaned, high-quality VQA dataset**

Properties:
- Visually grounded
- Medically consistent
- Domain-specific terminology

---

# 🧪 Model Fine-tuning

<!-- INSERT FIGURE: Phase 2 training pipeline -->
> 📌 Use: "Overview of Phase 2" (Only Attention vs Full Linear)

Base model:
- **Qwen2-VL-7B Instruct**

Fine-tuning method:
- **LoRA (Low-Rank Adaptation)**

Tested configurations:
- Only Attention layers
- Full Linear layers (**Final model: GalLens-Expert**)

Findings:
- Base model has **almost no poultry disease knowledge**
- Fine-tuned models learn:
  - Visual patterns
  - Medical terminology
  - Disease-specific features

---

# 📚 Retrieval-Augmented Generation (RAG)

<!-- INSERT FIGURE: RAG pipeline -->
> 📌 Use: Phase 3 diagram (Knowledge Base → Embedding → Vector DB → Retrieve → VLM)

Knowledge sources:
- Veterinary manuals
- Medical guidelines
- PDF documents
- Trusted websites

Pipeline:
1. Index documents using embedding model
2. Store in vector database
3. At query time:
   - Encode question
   - Retrieve relevant chunks
   - Inject into VLM prompt

Purpose:
- Reduce hallucination
- Improve factual correctness
- Provide treatment knowledge

---

# 🤖 Inference Pipeline

<!-- INSERT FIGURE: Inference routing logic -->
> 📌 You can reuse Phase 3 routing diagram

Two modes:

### 1️⃣ Visual Diagnosis
- Input: Image + "What disease is this?"
- Output: Disease name + visual explanation

### 2️⃣ Medical Consultation
- Input: "How to treat Newcastle disease?"
- Output: Retrieved knowledge + grounded explanation

---

# 📊 Experimental Results

<!-- INSERT FIGURE: Confusion Matrix - Base model -->
<!-- INSERT FIGURE: Confusion Matrix - Expert model -->
> 📌 Use: cm_base_model.png, cm_expert_model.png

Observations:
- Base model collapses predictions into "Other"
- Fine-tuned model separates visually similar diseases

<!-- INSERT FIGURE: Quantitative result table -->
> 📌 Use: Your metric comparison table

Metrics:
- ROUGE-L, BERT Similarity
- G-Eval (Accuracy, Relevance, Fluency)
- Accuracy, F1-score, Recall

Result:
- **GalLens-Expert performs best on all metrics**

---

# 📁 Project Structure

