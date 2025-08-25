"""
Lightweight RAG tool for querying Pinecone vectorstore.
Provides agents with retrieval-augmented generation capabilities.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGResult:
    """Result of RAG query."""
    query: str
    context: str
    sources: List[Dict[str, Any]]
    score_threshold: float


class RAGToolSchema(BaseModel):
    """Schema for RAG tool arguments."""
    query: str = Field(description="The query to search for in the knowledge base")
    top_k: int = Field(default=3, description="Number of top results to retrieve")
    score_threshold: float = Field(default=0.5, description="Minimum similarity score threshold")


class RAGTool(BaseTool):
    """
    Lightweight RAG tool for querying Pinecone vectorstore.
    
    This tool enables agents to retrieve relevant information from a knowledge base
    stored in Pinecone and provides it as context for decision making.
    """
    
    name: str = "retrieve_orchestration_knowledge"
    description: str = """Retrieve project management and orchestration knowledge from the knowledge base.
    
    This tool provides access to domain-specific information on project management that can help with:
    - planning strategies
    - Best practices
    - Project management methodologies and frameworks

    *IMPORTANT*: Keep queries as general as possible, as the knowledge comes from a project management handbook from the EU
    
    Input: Natural language query about orchestration, project management, or coordination
    Output: Relevant knowledge and context from the knowledge base
    
    Use this tool when you need guidance on how to structure tasks, coordinate agents, 
    or apply proven project management approaches to complex problems."""
    
    # Configuration
    pinecone_api_key: Optional[str] = Field(default=None, exclude=True)
    openai_api_key: Optional[str] = Field(default=None, exclude=True)
    index_name: str = Field(default="", exclude=True)
    
    # Clients
    pc: Optional[Pinecone] = Field(default=None, exclude=True)
    openai_client: Optional[openai.OpenAI] = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    retrieval_history: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get configuration from environment
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX", "mas-realm")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Set args_schema
        self.args_schema = RAGToolSchema
        
        # Initialize index lazily
        self.index = None
    
    def _ensure_index(self):
        """Ensure Pinecone index is initialized."""
        if self.index is None:
            try:
                self.index = self.pc.Index(self.index_name)
            except Exception as e:
                raise ValueError(f"Could not connect to Pinecone index '{self.index_name}': {e}")
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to generate query embedding: {e}")
    
    def _format_context(self, matches: List[Any]) -> str:
        """Format retrieved matches into context string."""
        if not matches:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, match in enumerate(matches, 1):
            text = match.metadata.get('text', 'No text available')
            source = match.metadata.get('filename', 'Unknown source')
            chunk_idx = match.metadata.get('chunk_index', '')
            
            source_info = f"{source}"
            if chunk_idx != '':
                source_info += f" (chunk {chunk_idx})"
            
            context_parts.append(f"[{i}] {text}\nSource: {source_info}\nScore: {match.score:.3f}")
        
        return "\n\n".join(context_parts)
    
    def _run(self, query: str, top_k: int = 3, score_threshold: float = 0.5, **kwargs) -> str:
        """Execute RAG query."""
        try:
            # Ensure index is initialized
            self._ensure_index()
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter by score threshold
            relevant_matches = [
                match for match in results.matches 
                if match.score >= score_threshold
            ]
            
            if not relevant_matches:
                # Still track the query even if no results
                self._track_knowledge_retrieval(query, [], top_k, score_threshold)
                return f"No relevant information found for query: '{query}' (minimum score: {score_threshold})"
            
            # Track the knowledge retrieval for metrics
            self._track_knowledge_retrieval(query, relevant_matches, top_k, score_threshold)
            
            # Format response
            context = self._format_context(relevant_matches)
            
            response = f"""Query: {query}

Retrieved Context:
{context}

Found {len(relevant_matches)} relevant document(s) with similarity scores above {score_threshold}."""
            
            return response
            
        except Exception as e:
            # Track failed query
            self._track_knowledge_retrieval(query, [], top_k, score_threshold, error=str(e))
            return f"RAG query failed: {str(e)}"
    
    async def _arun(self, query: str, top_k: int = 3, score_threshold: float = 0.5, **kwargs) -> str:
        """Async wrapper - just calls sync version."""
        return self._run(query, top_k, score_threshold, **kwargs)
    
    def _track_knowledge_retrieval(self, query: str, matches: List[Any], top_k: int, 
                                   score_threshold: float, error: Optional[str] = None) -> None:
        """Track knowledge retrieval for metrics and debugging."""
        import time
        
        retrieval_record = {
            "timestamp": time.time(),
            "query": query,
            "top_k_requested": top_k,
            "score_threshold": score_threshold,
            "matches_found": len(matches),
            "error": error
        }
        
        # Add detailed match information
        if matches and not error:
            retrieval_record["documents"] = []
            for match in matches:
                doc_info = {
                    "score": float(match.score),
                    "source": match.metadata.get('filename', 'Unknown'),
                    "chunk_index": match.metadata.get('chunk_index', ''),
                    "text_preview": match.metadata.get('text', '')[:200] + "..." if len(match.metadata.get('text', '')) > 200 else match.metadata.get('text', '')
                }
                retrieval_record["documents"].append(doc_info)
        else:
            retrieval_record["documents"] = []
        
        self.retrieval_history.append(retrieval_record)
    
    def get_retrieval_history(self) -> List[Dict[str, Any]]:
        """Get the history of all knowledge retrievals made by this tool instance."""
        return self.retrieval_history.copy()
    
    def clear_retrieval_history(self) -> None:
        """Clear the retrieval history."""
        self.retrieval_history.clear()


def create_rag_tool() -> RAGTool:
    """Create and return a RAG tool instance."""
    return RAGTool()