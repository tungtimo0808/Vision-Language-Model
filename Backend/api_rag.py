import os
import io
import base64
import logging
import gc
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = "/storage/student6/GalLens_student6"
CACHE_DIR_PATH = "/storage/student6/mustela_cache"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "final_lora_a100")

app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# import rag eingine
try:
    from rag_modules import rag_engine  
    HAS_RAG = True
    logger.info("RAG Engine loaded successfully")
except ImportError as e:
    HAS_RAG = False
    logger.warning(f"RAG Engine import failed: {e}. Running in standard mode.")

# load model
logger.info(f"Initializing on {DEVICE}...")
model = None
processor = None

try:
    logger.info("Loading Processor...")
    try:
        processor = AutoProcessor.from_pretrained(LORA_PATH, trust_remote_code=True)
    except:
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)

    logger.info("Loading Base Model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0"
    )

    if os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
        logger.info(f"Attaching LoRA Adapter from: {LORA_PATH}")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model.eval()
    else:
        model = base_model
        logger.warning("LoRA not found. Running Base Model.")

except Exception as e:
    logger.error(f"CRITICAL ERROR: {e}")
    model = None

# load helpers
def process_base64_image(image_input):
    try:
        if isinstance(image_input, str): 
            if 'base64,' in image_input:
                image_input = image_input.split('base64,')[1]
            image_bytes = base64.b64decode(image_input)
            img = Image.open(io.BytesIO(image_bytes))
        else: 
            img = Image.open(image_input)
        return img.convert('RGB')
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise ValueError("Invalid Image Format")

def generate_answer(image, question, model_type='finetuned', is_first_turn=False, disease_context=None):
    if not model: return None, "Model failed to load."
    
    try:
        prompt_text = question.strip()
        if not prompt_text: return None, "Please enter a question."

        # check is rag needed (using base model)
        needs_rag = False
        if HAS_RAG and rag_engine and model_type == 'finetuned':
            # Ask base model if this question needs medical RAG
            check_prompt = f"""Is this question about chicken disease diagnosis, treatment, symptoms, or prevention?
Question: "{prompt_text}"

Answer only "Yes" or "No"."""
            
            logger.info(f"Checking if RAG needed for: {prompt_text}")
            
            try:
                # Use base model to check
                check_conversation = [{"role": "user", "content": [{"type": "text", "text": check_prompt}]}]
                check_text = processor.apply_chat_template(check_conversation, tokenize=False, add_generation_prompt=True)
                check_inputs = processor(text=[check_text], padding=True, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    with model.disable_adapter():
                        check_output = model.generate(**check_inputs, max_new_tokens=10, do_sample=False)
                
                check_result = processor.batch_decode(
                    [out[len(inp):] for inp, out in zip(check_inputs.input_ids, check_output)],
                    skip_special_tokens=True
                )[0].strip().lower()
                
                needs_rag = 'yes' in check_result
                logger.info(f"{'RAG needed' if needs_rag else 'RAG not needed'} (base model said: {check_result})")
            except Exception as e:
                logger.warning(f"RAG check failed, defaulting to True: {e}")
                needs_rag = True

        # rag retrieval step
        rag_context = ""
        rag_warning = ""
        rag_result = {'results': [], 'is_reliable': False}
        
        # Only run RAG if needed
        if needs_rag:
            logger.info("Searching Knowledge Base...")
            
            # use confidence-aware search
            rag_result = rag_engine.search_with_confidence(
                prompt_text, k=3, disease_context=disease_context
            )
            
            if rag_result['results'] and rag_result['is_reliable']:
                # Reliable RAG results
                rag_context = "\n\nMEDICAL REFERENCE KNOWLEDGE:\n" + "\n".join([
                    f"- {doc}" for doc in rag_result['results']
                ])
                logger.info(f"RAG: {len(rag_result['results'])} docs (confidence: {rag_result['top_score']:.2f})")
            elif rag_result['results'] and not rag_result['is_reliable']:
                # Low confidence - warn model
                rag_warning = f"\n\nLow confidence RAG: {rag_result['top_score']:.2f}"
                logger.warning(f"Low confidence RAG: {rag_result['top_score']:.2f}")
            else:
                # No results
                logger.info("Empty RAG result (skipped or no matches).")

        # --- PROMPT CONSTRUCTION ---
        final_prompt = prompt_text
        
        # 1. Insert RAG Context with improved prompt wrapper
        # Check if RAG was needed and executed
        if needs_rag and rag_result.get('results'):
            # RAG found results - check confidence
            if not rag_result.get('is_reliable', True):
                # Low confidence: Ask model to apologize
                final_prompt = f"""
You are a veterinary AI assistant. You searched for medical information but the confidence score was too low to provide accurate guidance.

You MUST respond EXACTLY:
"I apologize, but I cannot find accurate medical documentation for this specific issue in my knowledge base. Please consult a veterinarian for proper guidance."

Do NOT make up any information.
"""
                logger.info("Low confidence RAG - using apology prompt")
            else:
                # High confidence: Use RAG context
                final_prompt = f"""
Based on the following medical reference information, answer the user's question accurately and concisely.

MEDICAL REFERENCE KNOWLEDGE:
{rag_context}

IMPORTANT:
- Answer ONLY based on the information above
- Be concise and clear
- Do NOT repeat words or phrases
- If information is incomplete, state what you know and advise consulting a veterinarian

User's question: {prompt_text}

Answer:
"""
                logger.info("High confidence RAG - using context")
        else:
            # No RAG needed or no results - use original prompt
            if needs_rag:
                logger.info("RAG was needed but returned no results - using original prompt")
            else:
                logger.info("RAG not needed - using original prompt")

        # 2. First Turn Logic - Ask for image description
        if is_first_turn:
            if needs_rag and rag_result.get('results') and not rag_result.get('is_reliable', True):
                # Low confidence medical query on first turn - keep apology prompt as is
                pass
            else:
                # Normal or high confidence - add description request
                original_question = prompt_text
                if needs_rag and rag_result.get('results') and rag_result.get('is_reliable', True):
                    # Has reliable RAG context
                    final_prompt = f"Describe this image in detail first. Then based on the medical reference knowledge above, answer: {original_question}"
                else:
                    # No RAG or RAG not needed
                    final_prompt = f"Describe this image in detail first. Then answer: {original_question}"

        logger.info(f"Final Prompt: {final_prompt[:150]}...")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": final_prompt},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=[text_prompt], padding=True, return_tensors="pt").to(DEVICE)

        output_ids = None
        with torch.no_grad():
            if model_type == 'base':
                with model.disable_adapter():
                    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return result, None

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return None, str(e)

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    torch.cuda.empty_cache()
    gc.collect() 
    
    try:
        data = request.json if request.is_json else request.form
        image_input = data.get('image') 
        if not image_input: return jsonify({'success': False, 'error': 'No image provided'}), 400

        question = data.get('question', '')
        model_type = data.get('model_type', 'finetuned')
        is_first_turn = data.get('is_first_turn', False)
        # optimization 4: get disease context from request
        disease_context = data.get('disease_context', None)

        try:
            pil_image = process_base64_image(image_input)
            answer, error = generate_answer(pil_image, question, model_type, is_first_turn, disease_context)
        except Exception as e:
            logger.error(f"Generate answer error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

        if error: return jsonify({'success': False, 'error': error}), 400
            
        return jsonify({'success': True, 'answer': answer, 'model_used': model_type})

    except Exception as e:
        logger.error(f"API analyze error: {e}")
        return jsonify({'success': False, 'error': "Server Internal Error"}), 500

@app.route('/api/extract_disease', methods=['POST'])
def extract_disease():
    """Extract disease name from model response using base Qwen (no LoRA)"""
    try:
        data = request.json
        response_text = data.get('response_text', '')
        
        if not response_text:
            return jsonify({'success': False, 'error': 'No response text provided'}), 400
        
        # disease list
        known_diseases = [
            "Bumblefoot", "Fowlpox", "CRD", "Chronic Respiratory Disease",
            "Scalyleg", "Coccidiosis", "Marek's Disease", "Newcastle Disease",
            "Infectious Bronchitis", "Avian Influenza", "HPAI", "LPAI",
            "Infectious Laryngotracheitis", "ILT", "Blackhead", "Histomoniasis"
        ]
        
        # Use base Qwen model (without LoRA) for text understanding
        extraction_prompt = f"""Given this veterinary diagnosis response, identify the EXACT disease name mentioned.

Response text:
{response_text}

Known diseases: {', '.join(known_diseases)}

Return ONLY the disease name from the list above, or 'Unknown' if no match.
Format: Just the disease name, nothing else."""

        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": extraction_prompt}],
            }
        ]

        text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Use base model WITHOUT LoRA for pure text understanding
            with model.disable_adapter():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=32,  # Short output
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        disease_name = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Clean up result
        disease_name = disease_name.replace('Disease name:', '').replace('disease:', '').strip()
        
        # Validate against known diseases
        disease_lower = disease_name.lower()
        matched_disease = None
        for known in known_diseases:
            if known.lower() in disease_lower or disease_lower in known.lower():
                matched_disease = known
                break
        
        if matched_disease:
            logger.info(f"Extracted disease: {matched_disease}")
            return jsonify({
                'success': True, 
                'disease': matched_disease,
                'raw_extraction': disease_name
            })
        else:
            logger.info(f"No disease match found: {disease_name}")
            return jsonify({
                'success': True, 
                'disease': None,
                'raw_extraction': disease_name
            })

    except Exception as e:
        logger.error(f"Disease extraction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)