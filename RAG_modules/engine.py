"""
Features:
- Hybrid Search (Semantic + BM25)
- PDF Processing
- Re-ranking for precision
- Ground truth evaluation
- Deduplication
- Monitoring & Logging
"""

import os
import json
import glob
import logging
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        # use semantic only if ensemble not available
        EnsembleRetriever = None

# PDF Processing
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    print("pip install pypdf first")

# Re-ranking 
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

from .config import (
    DOCS_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_DIR,
    TOP_K_RETRIEVAL, FINAL_TOP_K, USE_HYBRID_SEARCH, USE_RERANKING,
    RERANK_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PROCESS_PDF,
    LOG_RETRIEVAL_METRICS, METRICS_LOG_FILE,
    # Optimizations
    ENABLE_DISEASE_SCOPED_RETRIEVAL, ENABLE_FALLBACK_SEARCH,
    ENABLE_CONDITIONAL_RAG, ENABLE_QUERY_EXPANSION, ENABLE_CONFIDENCE_FILTERING,
    DIAGNOSIS_ONLY_PATTERNS, MIN_CONFIDENCE_SCORE, KNOWN_DISEASES
)

#logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEngineV2:
    """RAG Engine with Hybrid Search, PDF Support, Re-ranking"""
    
    def __init__(self):
        logger.info("⚙️ Initializing RAG Engine...")
        
        # 1. Setup Embedding
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=EMBEDDING_CACHE_DIR
        )
        logger.info(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
        
        # 2. Connect ChromaDB (Semantic Search)
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embedding_fn
        )
        
        # 3. Initialize BM25 Retriever (Keyword Search)
        self.bm25_retriever = None
        self.all_documents = []  # Cache for BM25
        
        # 4. Re-ranker
        self.reranker = None
        if USE_RERANKING and HAS_RERANKER:
            try:
                self.reranker = CrossEncoder(RERANK_MODEL)
                logger.info(f"✅ Re-ranker loaded: {RERANK_MODEL}")
            except Exception as e:
                logger.warning(f"⚠️ Re-ranker failed to load: {e}")
        
        # 5. Text splitter for PDF
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        ) if HAS_PDF_SUPPORT else None
        
        # 6. Metrics tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'cache_hits': 0
        }
        
        # 7. Query cache for performance
        self.query_cache = {}
        self.cache_max_size = 100
        
        try:
            count = self.vector_db._collection.count()
            logger.info(f"RAG Engine Ready. Docs in DB: {count}")
        except:
            logger.info("RAG Engine Initialized (Empty DB).")

    # data ingestion
    
    def ingest_data(self):
        """Ingest both JSON Knowledge Base and PDF documents"""
        logger.info(f"Scanning Knowledge Base in {DOCS_DIR}...")
        
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
            logger.warning("Folder knowledge_base created. Please upload data!")
            return

        documents = []
        
        # ingest JSON kb
        kb_folder = os.path.join(DOCS_DIR, "KB")
        if os.path.exists(kb_folder):
            json_files = glob.glob(os.path.join(kb_folder, "*.json"))
            logger.info(f"Found {len(json_files)} JSON files in KB/")
            
            for file_path in json_files:
                docs = self._load_json_knowledge(file_path)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} chunks from {os.path.basename(file_path)}")
        
        # ingest PDF files
        if PROCESS_PDF and HAS_PDF_SUPPORT:
            gt_folder = os.path.join(DOCS_DIR, "Ground-truth")
            if os.path.exists(gt_folder):
                pdf_files = glob.glob(os.path.join(gt_folder, "*.pdf"))
                logger.info(f"Found {len(pdf_files)} PDF files in Ground-truth/")
                
                for pdf_path in pdf_files:
                    pdf_docs = self._load_pdf_document(pdf_path)
                    documents.extend(pdf_docs)
                    logger.info(f"Loaded {len(pdf_docs)} chunks from {os.path.basename(pdf_path)}")
        
        if not documents:
            logger.warning("No valid documents found to ingest.")
            return

        logger.info(f"Preparing to index {len(documents)} knowledge units...")

        # Clear old database
        if os.path.exists(VECTOR_DB_DIR):
            logger.info("Clearing old database...")
            try:
                self.vector_db.delete_collection()
            except:
                pass
            self.vector_db = Chroma(
                persist_directory=VECTOR_DB_DIR, 
                embedding_function=self.embedding_fn
            )
        
        # Batch indexing to ChromaDB
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vector_db.add_documents(batch)
            logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

        # Update BM25 retriever with all documents
        self.all_documents = documents
        if USE_HYBRID_SEARCH:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = TOP_K_RETRIEVAL
                logger.info("BM25 retriever initialized for hybrid search")
            except Exception as e:
                logger.warning(f"BM25 initialization failed: {e}")
        
        logger.info(f"Data ingestion complete! Total indexed: {len(documents)} chunks")
    
    def _load_json_knowledge(self, file_path: str) -> List[Document]:
        """Load JSON knowledge base with enhanced metadata"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                doc = self._process_json_item(item, file_path)
                if doc:
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        
        return documents
    
    def _process_json_item(self, item: Dict, file_path: str) -> Optional[Document]:
        """Process single JSON item into Document"""
        content = item.get('content', '')
        if not content:
            return None

        # Clean metadata (ChromaDB doesn't accept lists)
        raw_metadata = item.get('metadata', {}).copy() if isinstance(item.get('metadata'), dict) else {}
        clean_metadata = {}
        
        for key, value in raw_metadata.items():
            if isinstance(value, list):
                clean_metadata[key] = ", ".join(str(v) for v in value)
            elif value is not None:
                clean_metadata[key] = value
        
        # Add source info
        clean_metadata['source'] = os.path.basename(file_path)
        clean_metadata['source_type'] = 'json_kb'
        clean_metadata['chunk_id'] = item.get('id', 'unknown')
        
        # optimi - extract disease from filename/metadata
        disease_tag = self._extract_disease_tag(file_path, content, raw_metadata)
        if disease_tag:
            clean_metadata['disease'] = disease_tag
        
        # Augment content with hypothetical questions
        vector_context = item.get('vector_context', {})
        hypothetical_questions = vector_context.get('hypothetical_questions', [])
        
        augmented_content = content
        if hypothetical_questions:
            augmented_content += "\n\nRelated Questions: " + "; ".join(hypothetical_questions)

        return Document(
            page_content=augmented_content, 
            metadata={**clean_metadata, "original_content": content} 
        )
    
    def _load_pdf_document(self, pdf_path: str) -> List[Document]:
        """Load and chunk PDF document"""
        documents = []
        
        if not HAS_PDF_SUPPORT or not self.text_splitter:
            logger.warning("⚠️ PDF support not available")
            return documents
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Enhance metadata
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': os.path.basename(pdf_path),
                    'source_type': 'pdf',
                    'chunk_id': f"pdf_{os.path.basename(pdf_path)}_{idx}",
                    'original_content': chunk.page_content
                })
                
                # optimi - tag PDF chunks with diseases
                disease_tag = self._extract_disease_tag(pdf_path, chunk.page_content, {})
                if disease_tag:
                    chunk.metadata['disease'] = disease_tag
                
                documents.append(chunk)
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
        
        return documents
    
    def _extract_disease_tag(self, file_path: str, content: str, metadata: Dict) -> Optional[str]:
        """
        optimi - extract disease tag from filename or content
        Used for disease-scoped retrieval
        """
        # Try filename first
        filename = os.path.basename(file_path).lower()
        for disease in KNOWN_DISEASES:
            if disease.lower().replace(' ', '_') in filename or disease.lower() in filename:
                return disease
        
        # Try metadata
        if 'disease' in metadata:
            return str(metadata['disease'])
        
        # Try content (simple keyword matching)
        content_lower = content.lower()
        for disease in KNOWN_DISEASES:
            if disease.lower() in content_lower:
                return disease
        
        return None

    # retrieval
    
    def search(self, query: str, k: Optional[int] = None, disease_context: Optional[str] = None, 
               return_scores: bool = False) -> List[str]:
        """
        Main search interface with all optimizations
        
        Args:
            query: User query
            k: Number of results to return (default: FINAL_TOP_K)
            disease_context: Current disease from conversation (for scoped retrieval)
            return_scores: Return (content, score) tuples instead of just content
        
        Returns:
            List of relevant content strings (or tuples if return_scores=True)
        """
        k = k or FINAL_TOP_K
        
        # optimi: Conditional RAG - Skip only for pure diagnosis
        if ENABLE_CONDITIONAL_RAG:
            # Check if pure diagnosis query (no knowledge request)
            is_pure_diagnosis = any(p in query.lower() for p in DIAGNOSIS_ONLY_PATTERNS)
            # But allow if also asking for info/treatment
            has_info_request = any(word in query.lower() for word in ['what is', 'how', 'why', 'treat', 'prevent', 'cause'])
            
            if is_pure_diagnosis and not has_info_request:
                logger.info(f"Pure diagnosis query, model can handle: {query[:50]}...")
                return []  # Skip RAG - model is good at diagnosis
        
        # optimi - Query Expansion
        if ENABLE_QUERY_EXPANSION:
            query = self._expand_query(query, disease_context)
        
        # optimi - Cache key includes disease_context
        cache_key = f"{query}_{k}_{disease_context or 'none'}"
        if cache_key in self.query_cache:
            self.retrieval_stats['cache_hits'] += 1
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        # Check if DB is empty
        try:
            if self.vector_db._collection.count() == 0:
                return []
        except:
            return []
        
        # optimi - Disease-Scoped Retrieval
        if ENABLE_DISEASE_SCOPED_RETRIEVAL and disease_context:
            docs = self._scoped_search(query, disease_context, TOP_K_RETRIEVAL)
            
            # optimi - Fallback to general search if empty
            if ENABLE_FALLBACK_SEARCH and len(docs) == 0:
                logger.info(f"Scoped search empty, falling back to general search")
                docs = self._general_search(query, TOP_K_RETRIEVAL)
        else:
            # General search (no disease context)
            docs = self._general_search(query, TOP_K_RETRIEVAL)
        
        # Re-rank if enabled
        if USE_RERANKING and self.reranker and len(docs) > 0:
            docs = self._rerank_documents(query, docs)
        
        # Deduplicate
        docs = self._deduplicate_documents(docs)
        
        # optimi - Confidence Filtering
        if ENABLE_CONFIDENCE_FILTERING and return_scores:
            # Need to get scores - use similarity_search_with_score
            docs_with_scores = self._get_docs_with_scores(query, docs, disease_context)
            docs_with_scores = [(doc, score) for doc, score in docs_with_scores 
                               if score >= MIN_CONFIDENCE_SCORE]
            docs = [doc for doc, _ in docs_with_scores]
        
        # Take top K
        docs = docs[:k]
        
        # Extract content
        results = [doc.metadata.get("original_content", doc.page_content) for doc in docs]
        
        # Update metrics
        elapsed = time.time() - start_time
        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['avg_retrieval_time'] = (
            (self.retrieval_stats['avg_retrieval_time'] * (self.retrieval_stats['total_queries'] - 1) + elapsed) 
            / self.retrieval_stats['total_queries']
        )
        
        # Cache result
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = results
        
        # Log
        if LOG_RETRIEVAL_METRICS:
            self._log_retrieval(query, len(results), elapsed, disease_context)
        
        logger.info(f"🔍 Retrieved {len(results)} docs for: {query[:50]}... (took {elapsed:.3f}s)")
        
        return results
    
    def search_with_confidence(self, query: str, k: Optional[int] = None,
                               disease_context: Optional[str] = None) -> dict:
        """
        Search and return confidence score for RAG quality control
        
        Returns:
            {
                'results': List[str],  # Retrieved chunks
                'top_score': float,    # Confidence of best result (0-1)
                'is_reliable': bool    # Whether RAG results are trustworthy
            }
        """
        from .config import RAG_CONFIDENCE_THRESHOLD
        
        k = k or FINAL_TOP_K
        
        # Same retrieval logic as search()
        if ENABLE_CONDITIONAL_RAG:
            is_pure_diagnosis = any(p in query.lower() for p in DIAGNOSIS_ONLY_PATTERNS)
            has_info_request = any(word in query.lower() for word in ['what is', 'how', 'why', 'treat', 'prevent', 'cause'])
            if is_pure_diagnosis and not has_info_request:
                return {'results': [], 'top_score': 0.0, 'is_reliable': False}
        
        # Retrieve docs
        if ENABLE_DISEASE_SCOPED_RETRIEVAL and disease_context:
            docs = self._scoped_search(query, disease_context, TOP_K_RETRIEVAL)
            if ENABLE_FALLBACK_SEARCH and len(docs) == 0:
                docs = self._general_search(query, TOP_K_RETRIEVAL)
        else:
            docs = self._general_search(query, TOP_K_RETRIEVAL)
        
        # Re-rank and get scores
        top_score = 0.0
        if USE_RERANKING and self.reranker and len(docs) > 0:
            # Get reranker scores
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            
            # Sort by score
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in doc_score_pairs]
            top_score = float(doc_score_pairs[0][1]) if doc_score_pairs else 0.0
        
        docs = docs[:k]
        results = [doc.metadata.get("original_content", doc.page_content) for doc in docs]
        
        is_reliable = top_score >= RAG_CONFIDENCE_THRESHOLD
        
        return {
            'results': results,
            'top_score': top_score,
            'is_reliable': is_reliable
        }
    
    def _semantic_search(self, query: str, k: int) -> List[Document]:
        """Pure semantic similarity search"""
        return self.vector_db.similarity_search(query, k=k)
    
    def _general_search(self, query: str, k: int) -> List[Document]:
        """General search without disease filtering"""
        if USE_HYBRID_SEARCH and self.bm25_retriever:
            return self._hybrid_search(query, k)
        else:
            return self._semantic_search(query, k)
    
    def _scoped_search(self, query: str, disease_context: str, k: int) -> List[Document]:
        """
        optimi - Disease-scoped retrieval
        Search only within chunks tagged with the disease
        """
        try:
            # Try to search with metadata filter
            if USE_HYBRID_SEARCH and self.bm25_retriever:
                # For BM25, manually filter after retrieval
                all_docs = self._hybrid_search(query, k * 3)  # Get more, then filter
                filtered_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get('disease', '').lower() == disease_context.lower()
                ]
                return filtered_docs[:k]
            else:
                # Semantic search with filter
                return self.vector_db.similarity_search(
                    query, 
                    k=k,
                    filter={"disease": disease_context}
                )
        except Exception as e:
            logger.warning(f"Scoped search failed: {e}, falling back to general")
            return self._general_search(query, k)
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """Hybrid search combining semantic + BM25"""
        # Fallback to semantic if EnsembleRetriever not available
        if EnsembleRetriever is None:
            logger.warning("EnsembleRetriever not available, using semantic search only")
            return self._semantic_search(query, k)
        
        try:
            # Create ensemble retriever
            semantic_retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
            
            ensemble = EnsembleRetriever(
                retrievers=[semantic_retriever, self.bm25_retriever],
                weights=[0.6, 0.4]  # 60% semantic, 40% keyword
            )
            
            return ensemble.invoke(query)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
            return self._semantic_search(query, k)
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not docs:
            return docs
        
        try:
            # Prepare pairs
            pairs = [[query, doc.page_content] for doc in docs]
            
            # Score
            scores = self.reranker.predict(pairs)
            
            # Sort by score
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in doc_score_pairs]
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return docs
    
    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        if not docs:
            return docs
        
        unique_docs = []
        seen_contents = set()
        
        for doc in docs:
            # Use first 100 chars as signature (simple dedup)
            signature = doc.page_content[:100].strip()
            if signature not in seen_contents:
                seen_contents.add(signature)
                unique_docs.append(doc)
        
        if len(unique_docs) < len(docs):
            logger.info(f"Deduplication: {len(docs)} → {len(unique_docs)} docs")
        
        return unique_docs
    
    def _expand_query(self, query: str, disease_context: Optional[str] = None) -> str:
        """
        optimi - Query Expansion
        Expand vague queries with disease context
        """
        query_lower = query.lower().strip()
        
        # Vague queries that need expansion
        vague_patterns = [
            'how to treat this', 'what to do', 'how to cure',
            'treatment', 'help', 'fix this', 'cure this'
        ]
        
        is_vague = any(pattern in query_lower for pattern in vague_patterns)
        
        if is_vague and disease_context:
            expanded = f"Treatment protocol for {disease_context}"
            logger.info(f"Query expanded: '{query}' → '{expanded}'")
            return expanded
        
        return query
    
    def _get_docs_with_scores(self, query: str, docs: List[Document], 
                               disease_context: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Get similarity scores for documents (for confidence filtering)"""
        try:
            if disease_context and ENABLE_DISEASE_SCOPED_RETRIEVAL:
                results = self.vector_db.similarity_search_with_score(
                    query, 
                    k=len(docs),
                    filter={"disease": disease_context}
                )
            else:
                results = self.vector_db.similarity_search_with_score(query, k=len(docs))
            return results
        except:
            # Fallback: assign neutral scores
            return [(doc, 0.7) for doc in docs]
    
    def _log_retrieval(self, query: str, num_results: int, elapsed: float, 
                       disease_context: Optional[str] = None):
        """Log retrieval metrics"""
        try:
            with open(METRICS_LOG_FILE, 'a', encoding='utf-8') as f:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query[:100],
                    'num_results': num_results,
                    'elapsed_time': round(elapsed, 4),
                    'disease_context': disease_context,
                    'cache_hit': False
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    # GT eval
    
    def evaluate_with_ground_truth(self, gt_file_path: str) -> Dict:
        logger.info(f"Evaluating with ground truth: {gt_file_path}")
        
        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return {}
        
        metrics = {
            'total_queries': len(ground_truth),
            'hit_at_1': 0,
            'hit_at_3': 0,
            'hit_at_5': 0,
            'mrr': 0.0,  # Mean Reciprocal Rank
            'failed_queries': []
        }
        
        for item in ground_truth:
            query = item.get('query', '')
            expected_ids = set(item.get('supporting_chunk_ids', []))
            
            if not query or not expected_ids:
                continue
            
            # Retrieve using internal method to get Document objects with metadata
            if USE_HYBRID_SEARCH and self.bm25_retriever:
                docs = self._hybrid_search(query, 10)  # Get more for re-ranking
            else:
                docs = self._semantic_search(query, 10)
            
            # re-rank if enabled (critical for evaluation accuracy)
            if USE_RERANKING and self.reranker and len(docs) > 0:
                docs = self._rerank_documents(query, docs)
            
            # Take top 5 for evaluation
            docs = docs[:5]
            
            # Extract chunk IDs from retrieved docs
            retrieved_ids = [doc.metadata.get('chunk_id', '') for doc in docs if doc.metadata.get('chunk_id')]
            
            # Calculate metrics
            for i, chunk_id in enumerate(retrieved_ids[:5]):
                if chunk_id in expected_ids:
                    if i == 0:
                        metrics['hit_at_1'] += 1
                    if i < 3:
                        metrics['hit_at_3'] += 1
                    metrics['hit_at_5'] += 1
                    
                    # MRR
                    metrics['mrr'] += 1.0 / (i + 1)
                    break
            else:
                metrics['failed_queries'].append({
                    'qid': item.get('qid'),
                    'query': query,
                    'expected': list(expected_ids),
                    'retrieved': retrieved_ids[:3]
                })
        
        # Normalize metrics
        total = metrics['total_queries']
        if total > 0:
            metrics['hit_at_1'] = round(metrics['hit_at_1'] / total, 4)
            metrics['hit_at_3'] = round(metrics['hit_at_3'] / total, 4)
            metrics['hit_at_5'] = round(metrics['hit_at_5'] / total, 4)
            metrics['mrr'] = round(metrics['mrr'] / total, 4)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"   Hit@1: {metrics['hit_at_1']:.2%}")
        logger.info(f"   Hit@3: {metrics['hit_at_3']:.2%}")
        logger.info(f"   Hit@5: {metrics['hit_at_5']:.2%}")
        logger.info(f"   MRR: {metrics['mrr']:.4f}")
        logger.info(f"   Failed: {len(metrics['failed_queries'])}/{total}")
        
        return metrics
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            **self.retrieval_stats,
            'db_size': self.vector_db._collection.count() if self.vector_db else 0,
            'cache_size': len(self.query_cache)
        }


# Global instance
rag_engine_v2 = RAGEngineV2()
