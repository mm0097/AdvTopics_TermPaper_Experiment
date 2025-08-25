import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class PineconeUpsertManager:
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        index_name: str = "default-index",
        dimension: int = 3072,  # text-embedding-3-large dimension
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.index = None
        
    def create_index_if_not_exists(self) -> None:
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully.")
            else:
                logger.info(f"Index '{self.index_name}' already exists.")
                
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error creating/accessing index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings using OpenAI text-embedding-3-large model."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting to avoid API limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        return embeddings
    
    def prepare_vectors(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Prepare vectors for upsert by generating embeddings."""
        texts = [doc.text for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vector = {
                "id": doc.id,
                "values": embedding,
                "metadata": doc.metadata or {}
            }
            # Add the original text to metadata for reference
            vector["metadata"]["text"] = doc.text
            vectors.append(vector)
        
        return vectors
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Upsert vectors to Pinecone index."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_index_if_not_exists() first.")
        
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                logger.info(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                
                self.index.upsert(vectors=batch)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error upserting batch {batch_num}: {e}")
                raise
        
        logger.info(f"Successfully upserted {len(vectors)} vectors to index '{self.index_name}'")
    
    def upsert_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """Complete upsert workflow: embed documents and upsert to Pinecone."""
        logger.info(f"Starting upsert process for {len(documents)} documents")
        
        # Ensure index exists
        if not self.index:
            self.create_index_if_not_exists()
        
        # Prepare vectors with embeddings
        vectors = self.prepare_vectors(documents)
        
        # Upsert to Pinecone
        self.upsert_vectors(vectors, batch_size)
        
        logger.info("Upsert process completed successfully")
    
    def query_similar(self, query_text: str, top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """Query the index for similar documents."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_index_if_not_exists() first.")
        
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query_text])[0]
        
        # Query the index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if not self.index:
            raise ValueError("Index not initialized.")
        
        return self.index.describe_index_stats()

def load_documents_from_json(file_path: str) -> List[Document]:
    """Load documents from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            id=item.get('id', str(len(documents))),
            text=item.get('text', ''),
            metadata=item.get('metadata', {})
        )
        documents.append(doc)
    
    return documents

def load_documents_from_markdown(file_path: str, chunk_size: int = 500, overlap: int = 75) -> List[Document]:
    """Load and chunk a markdown file into documents with overlap."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = []
    filename = os.path.basename(file_path)
    
    # Convert to character-based chunking for better overlap control
    text_length = len(content)
    chunks = []
    start = 0
    
    while start < text_length:
        # Determine end position
        end = start + chunk_size
        
        if end >= text_length:
            # Last chunk - take everything remaining
            chunk_text = content[start:]
        else:
            # Find a good break point (prefer sentence/paragraph boundaries)
            chunk_text = content[start:end]
            
            # Try to break at paragraph boundary first
            last_paragraph = chunk_text.rfind('\n\n')
            if last_paragraph > chunk_size * 0.5:  # Don't break too early
                chunk_text = content[start:start + last_paragraph]
                end = start + last_paragraph
            else:
                # Try to break at sentence boundary
                last_sentence = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )
                if last_sentence > chunk_size * 0.5:  # Don't break too early
                    chunk_text = content[start:start + last_sentence + 1]
                    end = start + last_sentence + 1
        
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text.strip(),
                'start': start,
                'end': end if end < text_length else text_length
            })
        
        # Move start position with overlap
        if end >= text_length:
            break
        start = end - overlap
    
    # Create documents from chunks
    for i, chunk_info in enumerate(chunks):
        doc = Document(
            id=f"{filename}_chunk_{i}",
            text=chunk_info['text'],
            metadata={
                'filename': filename,
                'source': 'markdown',
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_start': chunk_info['start'],
                'chunk_end': chunk_info['end'],
                'chunk_size': len(chunk_info['text']),
                'overlap_size': overlap
            }
        )
        documents.append(doc)
    
    return documents

def main():
    """CLI interface for upserting markdown files to Pinecone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upsert markdown file to Pinecone with OpenAI embeddings")
    parser.add_argument("markdown_file", help="Path to the markdown file to upsert")
    parser.add_argument("--index-name", default="markdown-index", help="Pinecone index name")
    parser.add_argument("--chunk-size", type=int, default=750, help="Text chunk size for splitting")
    parser.add_argument("--overlap", type=int, default=75, help="Overlap size between chunks (default: 100)")
    parser.add_argument("--query", help="Test query after upserting")
    
    args = parser.parse_args()
    
    try:
        # Load documents from markdown file
        logger.info(f"Loading markdown file: {args.markdown_file}")
        documents = load_documents_from_markdown(args.markdown_file, args.chunk_size, args.overlap)
        logger.info(f"Loaded {len(documents)} document chunks")
        
        # Initialize the manager
        manager = PineconeUpsertManager(
            index_name=args.index_name,
            dimension=3072
        )
        
        # Create index and upsert documents
        manager.create_index_if_not_exists()
        manager.upsert_documents(documents)
        
        # Test query if provided
        if args.query:
            logger.info(f"Testing query: {args.query}")
            query_results = manager.query_similar(args.query, top_k=3)
            
            print("\nTop matches:")
            for i, match in enumerate(query_results.matches, 1):
                print(f"{i}. Score: {match.score:.4f}")
                print(f"   ID: {match.id}")
                print(f"   Text: {match.metadata.get('text', 'N/A')[:200]}...")
                print()
        
        # Get index statistics
        stats = manager.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()