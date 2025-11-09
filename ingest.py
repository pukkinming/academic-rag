"""Ingest script for chunking, embedding, and upserting papers into Qdrant."""


import json
import uuid
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import tiktoken
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector

from config import settings
from models import Paper, Chunk, PaperSection

# Import fastembed for sparse vectors
try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("⚠ fastembed not installed. Hybrid search disabled. Install with: pip install fastembed")


class PaperIngestor:
    """Handles ingestion of papers into the vector database."""
    
    def __init__(self):
        """Initialize the ingestor with embedding model and Qdrant client."""
        print(f"Loading dense embedding model: {settings.embedding_model}")
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize sparse model for hybrid search
        self.sparse_model = None
        self.use_hybrid = settings.use_hybrid_search and FASTEMBED_AVAILABLE
        if self.use_hybrid:
            try:
                print(f"Loading sparse embedding model: {settings.sparse_model}")
                self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
                print("✓ Hybrid search enabled (Dense + Sparse BM25)")
            except Exception as e:
                print(f"⚠ Failed to load sparse model: {e}")
                print("  Falling back to dense-only search")
                self.use_hybrid = False
        else:
            print("✓ Dense-only search (hybrid search disabled)")
        
        print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        
        # Initialize tokenizer for chunk size estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if settings.qdrant_collection_name not in collection_names:
            print(f"Creating collection: {settings.qdrant_collection_name}")
            
            if self.use_hybrid:
                # Create collection with both dense and sparse vectors
                print("  Collection type: Hybrid (Dense + Sparse)")
                self.client.create_collection(
                    collection_name=settings.qdrant_collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
            else:
                # Create collection with dense vectors only (backward compatible)
                print("  Collection type: Dense only")
                self.client.create_collection(
                    collection_name=settings.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
        else:
            print(f"Collection already exists: {settings.qdrant_collection_name}")
            # TODO: Optionally verify collection schema matches expectations
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def _chunk_section(
        self,
        section: PaperSection,
        paper: Paper,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Chunk]:
        """
        Chunk a section into smaller pieces with overlap.
        
        Args:
            section: The section to chunk
            paper: The parent paper
            chunk_size: Target chunk size in tokens (default from settings)
            overlap: Overlap size in tokens (default from settings)
        
        Returns:
            List of Chunk objects
        """
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        text = section.text.strip()
        if not text:
            return []
        
        # Split into sentences (simple split on . ! ? followed by space or newline)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_tokens > chunk_size:
                if current_chunk:
                    # Save current chunk first
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        paper=paper,
                        section=section,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self._count_tokens(word)
                    if temp_tokens + word_tokens > chunk_size and temp_chunk:
                        chunk_text = " ".join(temp_chunk)
                        chunks.append(self._create_chunk(
                            text=chunk_text,
                            paper=paper,
                            section=section,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                        # Keep overlap
                        overlap_words = []
                        overlap_tokens = 0
                        for w in reversed(temp_chunk):
                            w_tokens = self._count_tokens(w)
                            if overlap_tokens + w_tokens <= overlap:
                                overlap_words.insert(0, w)
                                overlap_tokens += w_tokens
                            else:
                                break
                        temp_chunk = overlap_words + [word]
                        temp_tokens = overlap_tokens + word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                continue
            
            # Normal case: add sentence to current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    paper=paper,
                    section=section,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                # Keep last sentences that fit in overlap
                overlap_chunk = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self._count_tokens(sent)
                    if overlap_tokens + sent_tokens <= overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_chunk + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                text=chunk_text,
                paper=paper,
                section=section,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        paper: Paper,
        section: PaperSection,
        chunk_index: int
    ) -> Chunk:
        """Create a Chunk object from text and metadata."""
        # Generate deterministic chunk ID
        chunk_id = f"{paper.paper_id}_{section.section_name}_{chunk_index}"
        chunk_id = chunk_id.replace(" ", "_").replace("/", "_")
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            paper_id=paper.paper_id,
            title=paper.title,
            citation_pointer=paper.citation_pointer,
            section_name=section.section_name,
            page_start=section.page_start,
            page_end=section.page_end,
            year=paper.year,
            modality_tags=paper.modality_tags,
            chunk_index=chunk_index
        )
    
    def ingest_paper(self, paper: Paper) -> int:
        """
        Ingest a single paper by chunking, embedding, and upserting to Qdrant.
        
        Args:
            paper: Paper object to ingest
        
        Returns:
            Number of chunks created
        """
        print(f"\nIngesting paper: {paper.citation_pointer} - {paper.title}")
        
        # Chunk all sections
        all_chunks = []
        for section in paper.sections:
            section_chunks = self._chunk_section(section, paper)
            all_chunks.extend(section_chunks)
        
        print(f"  Created {len(all_chunks)} chunks")
        
        if not all_chunks:
            print("  No chunks to ingest")
            return 0
        
        # Generate embeddings
        texts = [chunk.text for chunk in all_chunks]
        
        # Dense embeddings
        print(f"  Generating dense embeddings...")
        dense_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=settings.embedding_batch_size
        )
        
        # Sparse embeddings (if hybrid mode)
        sparse_embeddings = None
        if self.use_hybrid and self.sparse_model:
            print(f"  Generating sparse embeddings (BM25)...")
            # FastEmbed returns a generator, convert to list
            sparse_embeddings = list(self.sparse_model.embed(texts))
        
        # Prepare points for Qdrant
        print(f"  Preparing points for upsert...")
        points = []
        for i, chunk in enumerate(all_chunks):
            payload = {
                "text": chunk.text,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "citation_pointer": chunk.citation_pointer,
                "section_name": chunk.section_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "year": chunk.year,
                "modality_tags": chunk.modality_tags or [],
                "chunk_index": chunk.chunk_index
            }
            
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            if self.use_hybrid and sparse_embeddings:
                # Hybrid mode: use named vectors
                # Convert FastEmbed SparseEmbedding to Qdrant SparseVector
                sparse_emb = sparse_embeddings[i]
                sparse_vector = SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                )
                
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": sparse_vector
                    },
                    payload=payload
                )
            else:
                # Dense-only mode: backward compatible
                point = PointStruct(
                    id=point_id,
                    vector=dense_embeddings[i].tolist(),
                    payload=payload
                )
            
            points.append(point)
        
        # Upsert to Qdrant
        print(f"  Upserting to Qdrant...")
        self.client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points
        )
        
        print(f"  ✓ Successfully ingested {len(all_chunks)} chunks")
        return len(all_chunks)
    
    def ingest_papers_from_json(self, json_path: Path, status_csv_path: Path = None) -> Dict[str, Any]:
        """
        Ingest papers from a JSON file or directory of JSON files.
        
        Args:
            json_path: Path to JSON file or directory containing JSON files
            status_csv_path: Optional path to save ingestion status CSV. 
                           If None, saves to 'ingestion_status_<timestamp>.csv' in current directory
        
        Returns:
            Dictionary with ingestion statistics including CSV path
        """
        json_path = Path(json_path)
        
        # Collect all JSON files to process
        json_files = []
        if json_path.is_file():
            json_files = [json_path]
        elif json_path.is_dir():
            json_files = sorted(json_path.glob("*.json"))
            if not json_files:
                print(f"No JSON files found in directory: {json_path}")
                return {
                    "papers_ingested": 0,
                    "total_chunks": 0,
                    "avg_chunks_per_paper": 0
                }
        else:
            raise ValueError(f"Path does not exist: {json_path}")
        
        print(f"Found {len(json_files)} JSON file(s) to process")
        
        # Setup CSV file for tracking ingestion status
        if status_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_csv_path = Path(f"ingestion_status_{timestamp}.csv")
        else:
            status_csv_path = Path(status_csv_path)
        
        print(f"📊 Saving ingestion status to: {status_csv_path}")
        
        # CSV headers
        csv_headers = [
            "json_file",
            "paper_id",
            "title",
            "citation_pointer",
            "status",
            "chunks_created",
            "error_message",
            "timestamp"
        ]
        
        total_chunks = 0
        papers_ingested = 0
        papers_failed = 0
        
        # Open CSV file for writing
        with open(status_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            
            for json_file in json_files:
                print(f"\nLoading paper from: {json_file.name}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle both single paper and list of papers
                    if isinstance(data, list):
                        papers_data = data
                    else:
                        papers_data = [data]
                    
                    for paper_data in papers_data:
                        paper_id = paper_data.get('paper_id', 'unknown')
                        title = paper_data.get('title', 'unknown')
                        citation = paper_data.get('citation_pointer', 'unknown')
                        timestamp = datetime.now().isoformat()
                        
                        try:
                            paper = Paper(**paper_data)
                            chunks_created = self.ingest_paper(paper)
                            total_chunks += chunks_created
                            papers_ingested += 1
                            
                            # Record success
                            writer.writerow([
                                json_file.name,
                                paper_id,
                                title[:100],  # Truncate long titles
                                citation,
                                "Success",
                                chunks_created,
                                "",
                                timestamp
                            ])
                            csvfile.flush()  # Ensure data is written immediately
                            
                        except Exception as e:
                            papers_failed += 1
                            error_msg = str(e)[:500]  # Truncate long error messages
                            print(f"  ✗ Error ingesting paper: {e}")
                            
                            # Record failure
                            writer.writerow([
                                json_file.name,
                                paper_id,
                                title[:100] if title else "unknown",
                                citation if citation else "unknown",
                                "Failed",
                                0,
                                error_msg,
                                timestamp
                            ])
                            csvfile.flush()  # Ensure data is written immediately
                            continue
                            
                except Exception as e:
                    papers_failed += 1
                    error_msg = str(e)[:500]
                    print(f"  ✗ Error loading JSON file {json_file.name}: {e}")
                    
                    # Record file loading failure
                    timestamp = datetime.now().isoformat()
                    writer.writerow([
                        json_file.name,
                        "unknown",
                        "unknown",
                        "unknown",
                        "Failed",
                        0,
                        f"File loading error: {error_msg}",
                        timestamp
                    ])
                    csvfile.flush()
                    continue
        
        stats = {
            "papers_ingested": papers_ingested,
            "papers_failed": papers_failed,
            "total_chunks": total_chunks,
            "avg_chunks_per_paper": total_chunks / papers_ingested if papers_ingested > 0 else 0,
            "status_csv_path": str(status_csv_path)
        }
        
        print(f"\n{'='*60}")
        print(f"Ingestion complete!")
        print(f"  Papers ingested: {papers_ingested}")
        print(f"  Papers failed: {papers_failed}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Avg chunks per paper: {stats['avg_chunks_per_paper']:.1f}")
        print(f"  Status CSV saved to: {status_csv_path}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main function for running the ingest script from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest academic papers into Qdrant vector database"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--status-csv",
        type=str,
        default=None,
        help="Optional path to save ingestion status CSV. If not provided, saves to 'ingestion_status_<timestamp>.csv'"
    )
    
    args = parser.parse_args()
    
    ingestor = PaperIngestor()
    csv_path = Path(args.status_csv) if args.status_csv else None
    ingestor.ingest_papers_from_json(Path(args.json_path), status_csv_path=csv_path)


if __name__ == "__main__":
    main()


