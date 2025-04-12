import logging
from typing import List, Dict, Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

log = logging.getLogger(__name__)

try:
    import bm25s
    import numpy as np
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    log.warning("bm25s package not found. BM25SRetriever will not be available.")


class BM25SRetriever(BaseRetriever):
    """Retriever that uses BM25s for retrieval."""
    
    texts: List[str] = Field(default_factory=list)  # Original texts
    metadatas: List[Dict[str, Any]] = Field(default_factory=list)  # Original metadatas
    k: int = 4  # Number of documents to retrieve
    
    _corpus_tokens: Any = PrivateAttr()
    _retriever: Any = PrivateAttr()
    
    def __init__(self, texts: List[str], metadatas: List[Dict[str, Any]], k: int = 4):
        """Initialize with texts and metadatas."""
        if not BM25S_AVAILABLE:
            raise ImportError(
                "bm25s package not found. Please install it with `pip install bm25s`."
            )
        
        # Initialize with Pydantic
        super().__init__(texts=texts, metadatas=metadatas, k=k)
        
        # Initialize private attributes
        self._corpus_tokens = bm25s.tokenize(texts)
        self._retriever = bm25s.BM25(method="lucene")
        self._retriever.index(self._corpus_tokens)
        
        # Optionally activate Numba scorer for better performance
        try:
            self._retriever.activate_numba_scorer()
            log.info("BM25S: Numba acceleration activated")
        except Exception as e:
            log.warning(f"BM25S: Could not activate Numba acceleration: {e}")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query."""
        # Tokenize the query
        query_tokens = bm25s.tokenize(query)
        
        # Log query information
        log.info(f"BM25S: Query tokenized to {len(query_tokens)} tokens")
        
        # Retrieve documents
        results, scores = self._retriever.retrieve(
            query_tokens, 
            corpus=self.texts, 
            k=self.k
        )
        
        # Log results information
        log.info(f"BM25S: Retrieved results shape: {results.shape}")
        log.info(f"BM25S: Retrieved scores shape: {scores.shape}")
        log.info(f"BM25S: Results type: {type(results)}")
        log.info(f"BM25S: Scores type: {type(scores)}")
        
        # Convert to Documents
        documents = []
        try:
            for i in range(results.shape[1]):
                doc_text = results[0, i]
                try:
                    doc_idx = self.texts.index(doc_text)
                    metadata = self.metadatas[doc_idx] if doc_idx < len(self.metadatas) else {}
                    # Convert numpy values to Python native types to ensure JSON serialization works
                    if isinstance(scores, np.ndarray) and i < len(scores[0]):
                        score_value = float(scores[0, i])
                        if 'score' not in metadata:
                            metadata['score'] = score_value
                    documents.append(Document(page_content=doc_text, metadata=metadata))
                except ValueError:
                    # If the text is not found in the list (shouldn't happen but just in case)
                    log.warning(f"BM25S: Document text not found in corpus: {doc_text[:100]}...")
                    metadata = {}
                    if isinstance(scores, np.ndarray) and i < len(scores[0]):
                        metadata['score'] = float(scores[0, i])
                    documents.append(Document(page_content=doc_text, metadata=metadata))
        except Exception as e:
            log.error(f"BM25S: Error processing results: {e}")
        
        return documents
