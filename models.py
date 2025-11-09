"""Data models for the RAG system."""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class PaperSection(BaseModel):
    """Represents a section from a paper."""
    section_name: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class Figure(BaseModel):
    """Represents a figure from a paper."""
    caption: str
    page: int


class Table(BaseModel):
    """Represents a table from a paper."""
    caption: str
    data: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # Optional - some tables may not have extracted data
    page: int


class Citation(BaseModel):
    """Represents a citation/reference from a paper."""
    citation_id: str
    title: str
    authors: Optional[List[str]] = Field(default_factory=list)  # Optional - some citations may not have authors
    year: Optional[str] = None
    venue: Optional[str] = None


class Paper(BaseModel):
    """Represents a complete academic paper."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    published: Optional[str] = None  # ISO format timestamp
    citation_pointer: Optional[str] = None  # e.g., "Narayanan et al., 2020"
    sections: List[PaperSection]
    keywords: Optional[List[str]] = None
    modality_tags: Optional[List[str]] = None  # e.g., ["gait", "facial", "speech"]
    venue: Optional[str] = None
    doi: Optional[str] = None
    figures: Optional[List[Figure]] = None
    tables: Optional[List[Table]] = None
    citations: Optional[List[Citation]] = None
    
    @model_validator(mode='after')
    def extract_year_and_citation(self):
        """Extract year from published field and generate citation_pointer if missing."""
        # Extract year from published field if year is not provided
        if self.year is None and self.published:
            try:
                # Parse ISO format timestamp (e.g., "2020-01-08T01:01:48Z")
                dt = datetime.fromisoformat(self.published.replace('Z', '+00:00'))
                self.year = dt.year
            except Exception:
                # Try just extracting the first 4 digits as year
                import re
                match = re.search(r'(\d{4})', self.published)
                if match:
                    self.year = int(match.group(1))
        
        # Generate citation_pointer if not provided
        if self.citation_pointer is None and self.authors and self.year:
            if len(self.authors) == 1:
                # Single author: "Smith, 2020"
                first_author_last = self.authors[0].split()[-1]
                self.citation_pointer = f"{first_author_last}, {self.year}"
            elif len(self.authors) == 2:
                # Two authors: "Smith and Jones, 2020"
                first_last = self.authors[0].split()[-1]
                second_last = self.authors[1].split()[-1]
                self.citation_pointer = f"{first_last} and {second_last}, {self.year}"
            else:
                # Three or more authors: "Smith et al., 2020"
                first_author_last = self.authors[0].split()[-1]
                self.citation_pointer = f"{first_author_last} et al., {self.year}"
        
        return self


class Chunk(BaseModel):
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    paper_id: str
    title: str
    citation_pointer: str
    section_name: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    year: int
    modality_tags: Optional[List[str]] = None
    chunk_index: int = 0  # Position of chunk within section


class RetrievedChunk(BaseModel):
    """Represents a retrieved chunk with score."""
    text: str
    title: str
    citation_pointer: str
    section_name: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    paper_id: str
    year: int
    score: float
    modality_tags: Optional[List[str]] = None


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    top_k: int = Field(default=8, ge=1, le=50)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    modality_tags: Optional[List[str]] = None


class Source(BaseModel):
    """Source citation information."""
    paper_id: str
    citation: str
    section: str
    pages: List[int]
    title: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response model for answered questions."""
    answer: str
    sources: List[Source]
    question: str


