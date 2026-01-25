import os

BASE_DIR = "/storage/student6/GalLens_student6"
DOCS_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db_store")

# embedding model 
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_CACHE_DIR = "/storage/student6/mustela_embedding_cache"

# retrieval config
TOP_K_RETRIEVAL = 10  # Retrieve candidates for re-ranking
FINAL_TOP_K = 3       # Final top-k to return
USE_HYBRID_SEARCH = True  # hybrid semantic + BM25
USE_RERANKING = True      # Cross-encoder re-ranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# rag quality control
RAG_CONFIDENCE_THRESHOLD = 0.3 

# chunking config
CHUNK_SIZE = 512      # Chunk size for PDF
CHUNK_OVERLAP = 50    # Overlap between chunks

# pdf processing
PROCESS_PDF = True    # Whether to process PDF
PDF_TABLE_STRATEGY = "hi_res"  # hi_res for tables

# optimization

ENABLE_DISEASE_SCOPED_RETRIEVAL = True  # Filter by disease context
ENABLE_FALLBACK_SEARCH = True          # Fallback if scoped returns empty


ENABLE_CONDITIONAL_RAG = True           # Only RAG for treatment queries
ENABLE_QUERY_EXPANSION = True           # Expand vague queries
ENABLE_CONFIDENCE_FILTERING = True      # Filter low-confidence results

# skip RAG for these
DIAGNOSIS_ONLY_PATTERNS = [
    'what disease', 'which disease', 'is this', 'is it',
    'does this look', 'looks like', 'seems like', 'could this be'
]

# Confidence threshold
MIN_CONFIDENCE_SCORE = 0.58  # Filter results below this score

# Disease names for metadata classification
KNOWN_DISEASES = [
    'Bumblefoot', 'Fowlpox', 'Avian Influenza', 'Newcastle Disease',
    'Chronic Respiratory Disease', 'Salmonella', 'Scaly Leg Mite',
    'Healthy', 'NCD', 'AI', 'CRD'
]

# monitoring
LOG_RETRIEVAL_METRICS = True  # Log retrieval performance
METRICS_LOG_FILE = os.path.join(BASE_DIR, "rag_metrics.log")