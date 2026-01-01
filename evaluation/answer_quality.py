"""Answer quality evaluation for the academic RAG system.

Evaluates answers on three dimensions:
- Faithfulness (0-1): Are all claims supported by retrieved chunks?
- Coverage (0-2): Did it cover key papers/angles?
- Utility (0-2): Is it publication-ready?

Also provides:
- Hallucination detection
- Citation verification
- Missing key papers identification
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from models import RetrievedChunk
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for answer quality metrics."""
    
    # Core metrics
    faithfulness_score: float = 0.0  # 0-1: Are claims supported by evidence?
    coverage_score: Optional[float] = None  # 0-2: Did it cover key papers/angles? (only when ground truth provided)
    utility_score: float = 0.0  # 0-2: Is it publication-ready?
    
    # Hallucination detection
    hallucination_count: int = 0
    potential_hallucinations: List[str] = field(default_factory=list)
    
    # Citation verification
    citation_verification: Dict[str, bool] = field(default_factory=dict)  # citation -> verified
    citations_found: List[str] = field(default_factory=list)
    citations_verified: List[str] = field(default_factory=list)
    citations_unverified: List[str] = field(default_factory=list)
    
    # Unsupported claims
    unsupported_claims: List[str] = field(default_factory=list)
    unsupported_claims_count: int = 0
    
    # Coverage details
    papers_cited: Set[str] = field(default_factory=set)
    papers_available: Set[str] = field(default_factory=set)
    missing_key_papers: List[str] = field(default_factory=list)
    missing_key_papers_count: int = 0
    citation_diversity_ratio: Optional[float] = None  # Informational metric when no ground truth
    
    # Utility details
    utility_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "faithfulness": round(self.faithfulness_score, 3),
            "coverage": round(self.coverage_score, 3) if self.coverage_score is not None else None,
            "utility": round(self.utility_score, 3),
            "hallucination_count": self.hallucination_count,
            "potential_hallucinations": self.potential_hallucinations,
            "citation_verification": self.citation_verification,
            "citations_found": self.citations_found,
            "citations_verified": self.citations_verified,
            "citations_unverified": self.citations_unverified,
            "unsupported_claims_count": self.unsupported_claims_count,
            "unsupported_claims": self.unsupported_claims[:5],  # Limit to first 5
            "papers_cited_count": len(self.papers_cited),
            "papers_available_count": len(self.papers_available),
            "citation_diversity_ratio": round(self.citation_diversity_ratio, 3) if self.citation_diversity_ratio is not None else None,
            "missing_key_papers": self.missing_key_papers,
            "missing_key_papers_count": self.missing_key_papers_count,
            "utility_breakdown": {k: round(v, 3) for k, v in self.utility_breakdown.items()},
        }


class AnswerQualityEvaluator:
    """Evaluates the quality of RAG-generated answers."""
    
    # Citation pattern: matches citation blocks including multiple citations
    # Examples: (Author et al., 2020), (Smith, 2019), (Wu et al., 2025; Lu et al., 2025)
    # Citation patterns
    CITATION_FULL_PARENS = r'\([^)]*[A-Z][a-z]+[^)]*\d{4}[^)]*\)'  # (Author et al., 2025)
    CITATION_YEAR_PARENS = r'[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?\s*\(\d{4}\)'  # Author et al. (2025)
    CITATION_PATTERN = rf'(?:{CITATION_FULL_PARENS}|{CITATION_YEAR_PARENS})'

    # More flexible citation pattern for matching chunks
    CITATION_FLEXIBLE_PATTERN = r'([A-Z][a-z]+)(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?,?\s*(\d{4})'
    
    # Academic vocabulary indicators
    ACADEMIC_TERMS = {
        "study", "research", "findings", "results", "analysis", "data",
        "methodology", "participants", "conducted", "investigated", "examined",
        "demonstrated", "revealed", "indicated", "suggests", "observed",
        "significant", "correlation", "hypothesis", "experiment", "literature",
        "framework", "approach", "model", "theory", "evidence", "conclusion",
        "implications", "limitations", "future", "proposed", "validated"
    }
    
    # Sentence starters that indicate claims without citations
    CLAIM_INDICATORS = [
        "studies have shown",
        "research indicates",
        "it has been demonstrated",
        "evidence suggests",
        "it is known that",
        "research has found",
        "scientists have discovered",
        "experiments show",
        "data indicates",
        "findings reveal",
    ]
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize the evaluator.
        
        Args:
            embedding_model: SentenceTransformer for semantic similarity.
                           If None, loads the model from settings.
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
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract citations from text, splitting multiple citations in parentheses.
        
        Args:
            text: The text to extract citations from
            
        Returns:
            List of individual citation strings (e.g., ["Wu et al., 2025", "Lu et al., 2025"])
        """
        # First, find all citation blocks (parenthesized groups)
        citation_blocks = re.findall(self.CITATION_PATTERN, text)
        
        citations = []
        
        for block in citation_blocks:
            # Remove outer parentheses if present
            block = block.strip()
            if block.startswith('(') and block.endswith(')'):
                block = block[1:-1].strip()
            
            # Split by semicolons to handle multiple citations
            parts = [p.strip() for p in block.split(';')]
            
            for part in parts:
                if not part:
                    continue
                
                # Must contain a year
                if re.search(r'\d{4}', part):
                    # Normalize the citation format
                    # Handle both formats: "Wu et al., 2025" and "Wu et al. (2025)"
                    citation = part.strip()
                    
                    # If it's in the format "Author (Year)", convert to "Author, Year"
                    year_paren_match = re.match(r'^(.+?)\s*\((\d{4})\)$', citation)
                    if year_paren_match:
                        author_part = year_paren_match.group(1).strip()
                        year = year_paren_match.group(2)
                        # Add comma if not present
                        if not author_part.endswith(','):
                            citation = f"{author_part}, {year}"
                        else:
                            citation = f"{author_part} {year}"
                    
                    citations.append(citation)
        
        return citations
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The text to split
            
        Returns:
            List of sentences
        """
        # Protect common abbreviations from being split
        # Replace periods in abbreviations with a placeholder
        protected = text
        abbreviations = [
            (r'\bet\s+al\.', 'et al†'),  # et al.
            (r'\bDr\.', 'Dr†'),
            (r'\bProf\.', 'Prof†'),
            (r'\bMr\.', 'Mr†'),
            (r'\bMs\.', 'Ms†'),
            (r'\bMrs\.', 'Mrs†'),
            (r'\bi\.e\.', 'i†e†'),
            (r'\be\.g\.', 'e†g†'),
            (r'\bvs\.', 'vs†'),
            (r'\bFig\.', 'Fig†'),
            (r'\bEq\.', 'Eq†'),
            (r'\bNo\.', 'No†'),
            (r'\bVol\.', 'Vol†'),
            (r'\bp\.', 'p†'),  # page
            (r'\bpp\.', 'pp†'),  # pages
        ]
        
        for pattern, replacement in abbreviations:
            protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)
        
        # Split on sentence boundaries (. ! ? followed by space and capital letter or end)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected)
        
        # Restore the periods in abbreviations
        restored = []
        for s in sentences:
            s = s.replace('†', '.')
            s = s.strip()
            if s:
                restored.append(s)
        
        return restored
    
    def verify_citation(
        self,
        citation: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a citation exists in the retrieved chunks.
        
        Args:
            citation: Citation string to verify (e.g., "Smith et al., 2020")
            chunks: List of retrieved chunks
            
        Returns:
            Tuple of (is_verified, matching_paper_id)
        """
        # Extract author name and year from citation
        match = re.search(self.CITATION_FLEXIBLE_PATTERN, citation)
        if not match:
            return False, None
        
        author_name = match.group(1).lower()
        year_str = match.group(2) if match.lastindex >= 2 else None
        year = int(year_str) if year_str else None
        
        # Check against chunk citation_pointers
        for chunk in chunks:
            pointer_lower = chunk.citation_pointer.lower()
            
            # Check if author name is in the citation pointer
            if author_name in pointer_lower:
                # If we have a year, also verify it matches
                if year is not None:
                    if str(year) in chunk.citation_pointer or chunk.year == year:
                        return True, chunk.paper_id
                else:
                    return True, chunk.paper_id
        
        return False, None
    
    def check_claim_support(
        self,
        claim: str,
        chunks: List[RetrievedChunk],
        similarity_threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Check if a claim is supported by the retrieved chunks.
        
        Uses semantic similarity to determine if the claim is grounded
        in the evidence.
        
        Args:
            claim: The claim sentence to check
            chunks: List of retrieved chunks
            similarity_threshold: Minimum similarity to consider supported
            
        Returns:
            Tuple of (is_supported, max_similarity_score)
        """
        if not chunks:
            return False, 0.0
        
        self._ensure_model_loaded()
        
        # Encode claim and chunk texts
        chunk_texts = [chunk.text for chunk in chunks]
        
        try:
            claim_embedding = self.embedding_model.encode(claim, convert_to_tensor=True)
            chunk_embeddings = self.embedding_model.encode(
                chunk_texts,
                convert_to_tensor=True,
                batch_size=settings.embedding_batch_size
            )
            
            # Normalize for cosine similarity
            claim_embedding = F.normalize(claim_embedding.unsqueeze(0), p=2, dim=1)
            chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
            
            # Compute similarities
            similarities = torch.mm(claim_embedding, chunk_embeddings.t()).squeeze(0)
            max_similarity = float(torch.max(similarities))
            
            return max_similarity >= similarity_threshold, max_similarity
            
        except Exception as e:
            logger.warning(f"Semantic similarity check failed: {e}")
            # Fallback to keyword overlap
            claim_words = set(claim.lower().split())
            for chunk in chunks:
                chunk_words = set(chunk.text.lower().split())
                overlap = len(claim_words & chunk_words) / max(len(claim_words), 1)
                if overlap > 0.3:  # 30% keyword overlap
                    return True, overlap
            return False, 0.0
    
    def evaluate_faithfulness(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
        similarity_threshold: float = 0.5
    ) -> Tuple[float, Dict[str, bool], List[str]]:
        """
        Evaluate faithfulness: Are all claims supported by retrieved evidence?
        
        Args:
            answer: The generated answer
            chunks: List of retrieved chunks
            similarity_threshold: Minimum similarity for claim support
            
        Returns:
            Tuple of (faithfulness_score, citation_verification, unsupported_claims)
        """
        # Extract and verify citations
        citations = self.extract_citations(answer)
        citation_verification = {}
        
        verified_papers = set()
        for citation in citations:
            is_verified, paper_id = self.verify_citation(citation, chunks)
            citation_verification[citation] = is_verified
            if is_verified and paper_id:
                verified_papers.add(paper_id)
        
        # Calculate citation verification ratio
        if citation_verification:
            citation_ratio = sum(citation_verification.values()) / len(citation_verification)
        else:
            citation_ratio = 0.5  # Neutral if no citations
        
        # Check for unsupported claims (sentences without citations)
        sentences = self.extract_sentences(answer)
        unsupported_claims = []
        supported_count = 0
        
        for sentence in sentences:
            # Skip very short sentences or structural elements
            if len(sentence.split()) < 5:
                supported_count += 1
                continue
            
            # Check if sentence has a citation
            has_citation = bool(re.search(self.CITATION_PATTERN, sentence))
            
            if has_citation:
                supported_count += 1
            else:
                # Check if content is grounded in evidence
                is_supported, similarity = self.check_claim_support(
                    sentence, chunks, similarity_threshold
                )
                if is_supported:
                    supported_count += 1
                else:
                    unsupported_claims.append(sentence)
        
        # Calculate sentence support ratio
        if sentences:
            sentence_ratio = supported_count / len(sentences)
        else:
            sentence_ratio = 0.5
        
        # Combine citation and sentence ratios
        # Weight citation verification more heavily (60%) than sentence grounding (40%)
        faithfulness_score = 0.6 * citation_ratio + 0.4 * sentence_ratio
        
        return faithfulness_score, citation_verification, unsupported_claims
    
    def evaluate_coverage(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
        ground_truth_papers: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Set[str], Set[str], List[str], Optional[float]]:
        """
        Evaluate coverage: Did it cover key papers/angles?
        
        Args:
            answer: The generated answer
            chunks: List of retrieved chunks
            ground_truth_papers: Optional list of expected paper citations
            
        Returns:
            Tuple of (coverage_score, papers_cited, papers_available, missing_key_papers, citation_diversity_ratio)
            - coverage_score: 0-2 scale (only when ground_truth_papers provided, None otherwise)
            - citation_diversity_ratio: Ratio of cited papers to available papers (informational, not a quality score)
        """
        # Get all unique papers from chunks
        papers_available = {chunk.citation_pointer for chunk in chunks}
        
        # Extract cited papers from answer
        citations = self.extract_citations(answer)
        papers_cited = set()
        
        for citation in citations:
            is_verified, paper_id = self.verify_citation(citation, chunks)
            if is_verified:
                # Find the citation_pointer for this paper
                for chunk in chunks:
                    if chunk.paper_id == paper_id:
                        papers_cited.add(chunk.citation_pointer)
                        break
        
        missing_key_papers = []
        coverage_score = None
        citation_diversity_ratio = None
        
        if ground_truth_papers:
            # With ground truth: calculate percentage of expected papers cited
            ground_truth_set = set(ground_truth_papers)
            cited_from_ground_truth = papers_cited & ground_truth_set
            coverage_ratio = len(cited_from_ground_truth) / len(ground_truth_set) if ground_truth_set else 0
            
            missing_key_papers = list(ground_truth_set - papers_cited)
            
            # Scale to 0-2 range
            coverage_score = coverage_ratio * 2.0
            coverage_score = min(2.0, max(0.0, coverage_score))
        else:
            # Without ground truth: calculate citation diversity ratio (informational only)
            if len(papers_available) > 0:
                citation_diversity_ratio = len(papers_cited) / len(papers_available)
            else:
                citation_diversity_ratio = 0.0
        
        return coverage_score, papers_cited, papers_available, missing_key_papers, citation_diversity_ratio
    
    def evaluate_utility(
        self,
        answer: str,
        question: str,
        chunks: List[RetrievedChunk],
        use_llm: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate utility: Is it publication-ready?
        
        Uses heuristics to assess:
        - Academic tone and vocabulary
        - Structure and organization
        - Completeness (appropriate length)
        - Citation density
        
        Args:
            answer: The generated answer
            question: The original question
            chunks: List of retrieved chunks
            use_llm: Whether to use LLM-as-judge (not implemented, uses heuristics)
            
        Returns:
            Tuple of (utility_score, breakdown)
            Score is 0-2 scale
        """
        breakdown = {}
        
        # 1. Citation density (0-0.5)
        # Good academic writing cites sources regularly
        citations = self.extract_citations(answer)
        sentences = self.extract_sentences(answer)
        
        if sentences:
            citation_density = len(citations) / len(sentences)
            # Ideal: 0.3-0.5 citations per sentence
            if citation_density >= 0.3:
                citation_score = 0.5
            elif citation_density >= 0.15:
                citation_score = 0.35
            elif citation_density > 0:
                citation_score = 0.2
            else:
                citation_score = 0.0
        else:
            citation_score = 0.0
        
        breakdown["citation_density"] = citation_score
        
        # 2. Length appropriateness (0-0.5)
        # Academic responses should be substantial but not excessive
        word_count = len(answer.split())
        
        if 200 <= word_count <= 600:
            length_score = 0.5
        elif 100 <= word_count < 200 or 600 < word_count <= 800:
            length_score = 0.35
        elif 50 <= word_count < 100 or 800 < word_count <= 1000:
            length_score = 0.2
        else:
            length_score = 0.1
        
        breakdown["length"] = length_score
        
        # 3. Academic vocabulary (0-0.5)
        # Use of academic terms and phrases
        answer_words = set(answer.lower().split())
        academic_word_count = len(answer_words & self.ACADEMIC_TERMS)
        academic_ratio = academic_word_count / max(len(answer_words), 1)
        
        if academic_ratio >= 0.05:
            vocab_score = 0.5
        elif academic_ratio >= 0.03:
            vocab_score = 0.35
        elif academic_ratio >= 0.01:
            vocab_score = 0.2
        else:
            vocab_score = 0.1
        
        breakdown["academic_vocabulary"] = vocab_score
        
        # 4. Structure indicators (0-0.5)
        # Check for organized structure: paragraphs, transitions, etc.
        structure_score = 0.0
        
        # Multiple paragraphs
        paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            structure_score += 0.2
        elif len(paragraphs) == 1 and len(sentences) >= 5:
            structure_score += 0.1
        
        # Transition words
        transition_words = [
            "however", "furthermore", "moreover", "additionally", "consequently",
            "therefore", "in contrast", "specifically", "notably", "importantly",
            "first", "second", "finally", "in conclusion", "overall"
        ]
        answer_lower = answer.lower()
        transition_count = sum(1 for word in transition_words if word in answer_lower)
        if transition_count >= 3:
            structure_score += 0.2
        elif transition_count >= 1:
            structure_score += 0.1
        
        # Has an introduction and conclusion feel
        if any(answer_lower.startswith(start) for start in ["the ", "this ", "in ", "recent "]):
            structure_score += 0.05
        if any(phrase in answer_lower for phrase in ["in conclusion", "overall", "in summary", "to summarize"]):
            structure_score += 0.05
        
        breakdown["structure"] = min(0.5, structure_score)
        
        # Calculate total utility score (0-2)
        utility_score = sum(breakdown.values()) * (2.0 / 2.0)  # Scale to 0-2
        
        # Clamp to [0, 2]
        utility_score = min(2.0, max(0.0, utility_score))
        
        return utility_score, breakdown
    
    def evaluate_comprehensive(
        self,
        answer: str,
        question: str,
        chunks: List[RetrievedChunk],
        ground_truth_papers: Optional[List[str]] = None
    ) -> QualityMetrics:
        """
        Perform comprehensive quality evaluation.
        
        Args:
            answer: The generated answer
            question: The original question
            chunks: List of retrieved chunks
            ground_truth_papers: Optional list of expected paper citations
            
        Returns:
            QualityMetrics with all evaluation results
        """
        metrics = QualityMetrics()
        
        # Faithfulness evaluation
        faithfulness, citation_verification, unsupported_claims = self.evaluate_faithfulness(
            answer, chunks
        )
        metrics.faithfulness_score = faithfulness
        metrics.citation_verification = citation_verification
        metrics.citations_found = list(citation_verification.keys())
        metrics.citations_verified = [c for c, v in citation_verification.items() if v]
        metrics.citations_unverified = [c for c, v in citation_verification.items() if not v]
        metrics.unsupported_claims = unsupported_claims
        metrics.unsupported_claims_count = len(unsupported_claims)
        
        # Coverage evaluation
        coverage, papers_cited, papers_available, missing_papers, citation_diversity_ratio = self.evaluate_coverage(
            answer, chunks, ground_truth_papers
        )
        metrics.coverage_score = coverage
        metrics.papers_cited = papers_cited
        metrics.papers_available = papers_available
        metrics.missing_key_papers = missing_papers
        metrics.missing_key_papers_count = len(missing_papers)
        metrics.papers_cited_len = len(papers_cited)
        metrics.papers_available_len = len(papers_available)
        metrics.citation_diversity_ratio = citation_diversity_ratio
        
        # Utility evaluation
        utility, utility_breakdown = self.evaluate_utility(answer, question, chunks)
        metrics.utility_score = utility
        metrics.utility_breakdown = utility_breakdown
        
        # Hallucination detection - reuse unsupported_claims from faithfulness evaluation
        # (No need to re-check - we already did semantic similarity in evaluate_faithfulness)
        metrics.potential_hallucinations = unsupported_claims
        metrics.hallucination_count = len(unsupported_claims)
        
        return metrics


# Convenience function for quick evaluation
def evaluate_answer(
    answer: str,
    question: str,
    chunks: List[RetrievedChunk],
    ground_truth_papers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function for answer quality evaluation.
    
    Args:
        answer: The generated answer
        question: The original question
        chunks: List of retrieved chunks
        ground_truth_papers: Optional list of expected paper citations
        
    Returns:
        Dictionary with quality metrics
    """
    evaluator = AnswerQualityEvaluator()
    metrics = evaluator.evaluate_comprehensive(
        answer, question, chunks, ground_truth_papers
    )
    return metrics.to_dict()

