"""Answer validation for the academic RAG system.

Validates answers for correctness:
- Citation verification: Check all citations exist in retrieved chunks
- Citation format validation: Ensure proper format (Author et al., Year)
- Missing citation detection: Find claims without citations
- Hallucination detection: Use semantic similarity to detect unsupported claims
- Confidence scoring: Overall confidence in answer validity (0-1)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from models import RetrievedChunk
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for answer validation results."""
    
    # Overall validity
    is_valid: bool = True
    confidence_score: float = 1.0  # 0-1: Overall confidence in answer validity
    
    # Citation validation
    citation_errors: List[str] = field(default_factory=list)
    citation_warnings: List[str] = field(default_factory=list)
    citations_total: int = 0
    citations_verified: int = 0
    citations_format_valid: int = 0
    
    # Missing citations
    missing_citations: List[str] = field(default_factory=list)
    missing_citations_count: int = 0
    
    # Hallucination detection
    hallucination_warnings: List[str] = field(default_factory=list)
    hallucination_count: int = 0
    
    # Detailed verification results
    citation_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "confidence": round(self.confidence_score, 3),
            "citation_errors": self.citation_errors,
            "citation_warnings": self.citation_warnings,
            "citations_total": self.citations_total,
            "citations_verified": self.citations_verified,
            "citations_format_valid": self.citations_format_valid,
            "missing_citations_count": self.missing_citations_count,
            "missing_citations": self.missing_citations[:5],  # Limit output
            "hallucination_warnings": self.hallucination_warnings[:5],
            "hallucination_count": self.hallucination_count,
        }


class AnswerValidator:
    """Validates RAG-generated answers for correctness."""
    
    # Standard academic citation patterns
    CITATION_PATTERNS = [
        # (Author et al., 2020)
        r'\(([A-Z][a-z]+)\s+et\s+al\.?,?\s*(\d{4})\)',
        # (Author, 2020)
        r'\(([A-Z][a-z]+),?\s*(\d{4})\)',
        # (Author and Author, 2020)
        r'\(([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+),?\s*(\d{4})\)',
        # (Author & Author, 2020)
        r'\(([A-Z][a-z]+)\s*&\s*([A-Z][a-z]+),?\s*(\d{4})\)',
    ]
    
    # Combined pattern for extracting any citation
    CITATION_EXTRACT_PATTERN = r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+|&\s*[A-Z][a-z]+))?(?:,?\s*\d{4})?)\)'
    
    # Patterns that indicate factual claims needing citations
    CLAIM_PATTERNS = [
        r'studies\s+(?:have\s+)?show(?:n|s)?',
        r'research\s+(?:has\s+)?(?:indicate[sd]?|suggest[sd]?|found|demonstrate[sd]?)',
        r'(?:it\s+)?has\s+been\s+(?:shown|demonstrated|found|observed)',
        r'evidence\s+suggests?',
        r'(?:scientists|researchers)\s+(?:have\s+)?(?:discovered|found|shown)',
        r'according\s+to\s+(?:recent\s+)?(?:studies|research)',
        r'data\s+(?:shows?|indicates?|suggests?)',
        r'findings?\s+(?:reveal[sd]?|indicate[sd]?|suggest[sd]?)',
        r'\d+%\s+(?:of|accuracy|precision|recall)',  # Statistics
        r'significant(?:ly)?\s+(?:higher|lower|more|less|better|worse)',
    ]
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize the validator.
        
        Args:
            embedding_model: SentenceTransformer for semantic similarity.
                           If None, loads lazily when needed.
        """
        self.embedding_model = embedding_model
        self._model_loaded = False
    
    def _ensure_model_loaded(self):
        """Lazy load the embedding model."""
        if not self._model_loaded:
            if self.embedding_model is None:
                device = 'cuda' if settings.use_gpu and torch.cuda.is_available() else 'cpu'
                self.embedding_model = SentenceTransformer(settings.embedding_model, device=device)
            self._model_loaded = True
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citations from text with metadata.
        
        Args:
            text: The text to extract citations from
            
        Returns:
            List of citation dictionaries with keys: raw, author, year, format_valid
        """
        citations = []
        
        # Find all citation-like patterns
        matches = re.finditer(self.CITATION_EXTRACT_PATTERN, text)
        
        for match in matches:
            raw = match.group(1)
            
            # Try to parse author and year
            author = None
            year = None
            format_valid = False
            
            # Check against each pattern
            for pattern in self.CITATION_PATTERNS:
                sub_match = re.match(pattern.replace(r'\(', '').replace(r'\)', ''), '(' + raw + ')')
                if sub_match:
                    groups = sub_match.groups()
                    author = groups[0]
                    year = int(groups[-1]) if groups[-1] else None
                    format_valid = True
                    break
            
            # Fallback: try to extract year from raw
            if year is None:
                year_match = re.search(r'(\d{4})', raw)
                if year_match:
                    year = int(year_match.group(1))
            
            # Fallback: try to extract author from raw
            if author is None:
                author_match = re.search(r'^([A-Z][a-z]+)', raw)
                if author_match:
                    author = author_match.group(1)
            
            citations.append({
                'raw': raw,
                'author': author,
                'year': year,
                'format_valid': format_valid,
                'position': match.start()
            })
        
        return citations
    
    def verify_citation(
        self,
        citation: Dict[str, Any],
        chunks: List[RetrievedChunk]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify a citation against retrieved chunks.
        
        Args:
            citation: Citation dict from extract_citations
            chunks: List of retrieved chunks
            
        Returns:
            Tuple of (is_verified, matching_paper_id, matching_citation_pointer)
        """
        author = citation.get('author', '').lower() if citation.get('author') else ''
        year = citation.get('year')
        
        if not author:
            return False, None, None
        
        for chunk in chunks:
            pointer_lower = chunk.citation_pointer.lower()
            
            # Check if author name is in the citation pointer
            if author in pointer_lower:
                # If we have a year, also verify it
                if year is not None:
                    if str(year) in chunk.citation_pointer or chunk.year == year:
                        return True, chunk.paper_id, chunk.citation_pointer
                else:
                    return True, chunk.paper_id, chunk.citation_pointer
        
        return False, None, None
    
    def validate_citation_format(self, citation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate citation format.
        
        Args:
            citation: Citation dict from extract_citations
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        raw = citation.get('raw', '')
        
        # Check if it has the expected format
        if citation.get('format_valid'):
            return True, None
        
        # Check for common issues
        if not citation.get('year'):
            return False, f"Citation '{raw}' missing year"
        
        if not citation.get('author'):
            return False, f"Citation '{raw}' missing author name"
        
        # Year should be reasonable (1900-2100)
        year = citation.get('year')
        if year and (year < 1900 or year > 2100):
            return False, f"Citation '{raw}' has invalid year: {year}"
        
        return True, None
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def find_missing_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> List[str]:
        """
        Find sentences that make claims but lack citations.
        
        Args:
            answer: The generated answer
            chunks: List of retrieved chunks
            
        Returns:
            List of sentences that should have citations
        """
        sentences = self.extract_sentences(answer)
        missing = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
            
            # Skip if sentence has a citation
            if re.search(self.CITATION_EXTRACT_PATTERN, sentence):
                continue
            
            # Check if sentence makes a claim that needs citation
            is_claim = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in self.CLAIM_PATTERNS
            )
            
            if is_claim:
                missing.append(sentence)
        
        return missing
    
    def detect_hallucinations(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
        similarity_threshold: float = 0.4
    ) -> List[str]:
        """
        Detect potentially hallucinated content.
        
        Uses semantic similarity to find claims not grounded in evidence.
        
        Args:
            answer: The generated answer
            chunks: List of retrieved chunks
            similarity_threshold: Minimum similarity to consider grounded
            
        Returns:
            List of potentially hallucinated sentences
        """
        if not chunks:
            return []
        
        self._ensure_model_loaded()
        
        sentences = self.extract_sentences(answer)
        hallucinations = []
        
        # Patterns for specific factual claims
        fact_patterns = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\b\d+(?:\.\d+)?\s*(?:participants|subjects|samples|patients|users)',  # Sample sizes
            r'(?:accuracy|precision|recall|F1|AUC)\s*(?:of|:)?\s*\d+',  # Metrics
            r'\b\d+\s*(?:studies|papers|articles)',  # Study counts
            r'(?:up to|over|more than|less than)\s*\d+',  # Specific quantities
        ]
        
        try:
            # Encode all chunk texts
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(
                chunk_texts,
                convert_to_tensor=True,
                batch_size=settings.embedding_batch_size
            )
            chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
            
            for sentence in sentences:
                # Skip sentences with citations
                if re.search(self.CITATION_EXTRACT_PATTERN, sentence):
                    continue
                
                # Skip very short sentences
                if len(sentence.split()) < 6:
                    continue
                
                # Check if contains specific facts
                has_specific_facts = any(
                    re.search(pattern, sentence)
                    for pattern in fact_patterns
                )
                
                if has_specific_facts:
                    # Check semantic similarity
                    sentence_embedding = self.embedding_model.encode(
                        sentence, convert_to_tensor=True
                    )
                    sentence_embedding = F.normalize(sentence_embedding.unsqueeze(0), p=2, dim=1)
                    
                    similarities = torch.mm(sentence_embedding, chunk_embeddings.t()).squeeze(0)
                    max_similarity = float(torch.max(similarities))
                    
                    if max_similarity < similarity_threshold:
                        hallucinations.append(sentence)
        
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {e}")
            # Fallback to simple keyword matching
            for sentence in sentences:
                if not re.search(self.CITATION_EXTRACT_PATTERN, sentence):
                    if any(re.search(p, sentence) for p in fact_patterns):
                        # Check keyword overlap with chunks
                        sentence_words = set(sentence.lower().split())
                        max_overlap = 0
                        for chunk in chunks:
                            chunk_words = set(chunk.text.lower().split())
                            overlap = len(sentence_words & chunk_words) / max(len(sentence_words), 1)
                            max_overlap = max(max_overlap, overlap)
                        
                        if max_overlap < 0.2:  # Less than 20% keyword overlap
                            hallucinations.append(sentence)
        
        return hallucinations
    
    def calculate_confidence(
        self,
        citations_total: int,
        citations_verified: int,
        citations_format_valid: int,
        missing_citations_count: int,
        hallucination_count: int,
        answer_length: int
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            citations_total: Total number of citations found
            citations_verified: Number of verified citations
            citations_format_valid: Number of properly formatted citations
            missing_citations_count: Number of claims without citations
            hallucination_count: Number of potential hallucinations
            answer_length: Word count of the answer
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Citation verification penalty
        if citations_total > 0:
            verification_ratio = citations_verified / citations_total
            confidence *= (0.5 + 0.5 * verification_ratio)
        else:
            # No citations at all is concerning for academic writing
            confidence *= 0.6
        
        # Format validity penalty
        if citations_total > 0:
            format_ratio = citations_format_valid / citations_total
            confidence *= (0.8 + 0.2 * format_ratio)
        
        # Missing citations penalty
        # Normalize by expected claims based on answer length
        expected_claims = max(answer_length // 50, 3)  # Roughly 1 citation per 50 words
        if missing_citations_count > 0:
            penalty = min(0.3, missing_citations_count * 0.05)
            confidence *= (1 - penalty)
        
        # Hallucination penalty (severe)
        if hallucination_count > 0:
            penalty = min(0.4, hallucination_count * 0.1)
            confidence *= (1 - penalty)
        
        return max(0.0, min(1.0, confidence))
    
    def validate_comprehensive(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> ValidationResult:
        """
        Perform comprehensive answer validation.
        
        Args:
            answer: The generated answer
            chunks: List of retrieved chunks
            
        Returns:
            ValidationResult with all validation results
        """
        result = ValidationResult()
        
        # Extract and validate citations
        citations = self.extract_citations(answer)
        result.citations_total = len(citations)
        
        citation_details = {}
        
        for citation in citations:
            raw = citation['raw']
            
            # Check format
            format_valid, format_error = self.validate_citation_format(citation)
            if format_valid:
                result.citations_format_valid += 1
            else:
                result.citation_warnings.append(format_error)
            
            # Verify against chunks
            is_verified, paper_id, pointer = self.verify_citation(citation, chunks)
            if is_verified:
                result.citations_verified += 1
            else:
                result.citation_errors.append(f"Cannot verify citation: {raw}")
            
            citation_details[raw] = {
                'format_valid': format_valid,
                'verified': is_verified,
                'paper_id': paper_id,
                'matched_pointer': pointer
            }
        
        result.citation_details = citation_details
        
        # Find missing citations
        result.missing_citations = self.find_missing_citations(answer, chunks)
        result.missing_citations_count = len(result.missing_citations)
        
        # Detect hallucinations
        result.hallucination_warnings = self.detect_hallucinations(answer, chunks)
        result.hallucination_count = len(result.hallucination_warnings)
        
        # Calculate confidence
        answer_length = len(answer.split())
        result.confidence_score = self.calculate_confidence(
            result.citations_total,
            result.citations_verified,
            result.citations_format_valid,
            result.missing_citations_count,
            result.hallucination_count,
            answer_length
        )
        
        # Determine overall validity
        # Invalid if: more than 50% citations unverified, or 3+ hallucinations, or confidence < 0.5
        if result.citations_total > 0:
            verification_ratio = result.citations_verified / result.citations_total
            if verification_ratio < 0.5:
                result.is_valid = False
        
        if result.hallucination_count >= 3:
            result.is_valid = False
        
        if result.confidence_score < 0.5:
            result.is_valid = False
        
        return result


# Convenience function
def validate_answer(
    answer: str,
    chunks: List[RetrievedChunk]
) -> Dict[str, Any]:
    """
    Convenience function for answer validation.
    
    Args:
        answer: The generated answer
        chunks: List of retrieved chunks
        
    Returns:
        Dictionary with validation results
    """
    validator = AnswerValidator()
    result = validator.validate_comprehensive(answer, chunks)
    return result.to_dict()

