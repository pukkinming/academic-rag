"""Prompt builder for constructing academic-style prompts with proper citations."""

from typing import List, Dict, Any
from models import RetrievedChunk


SYSTEM_INSTRUCTION = """You are an expert in academic research and literature review.

Your task is to answer questions based ONLY on the provided evidence from academic papers. Follow these strict rules:

1. CITATION REQUIREMENT: Every factual claim MUST include an in-text citation in the format (Author et al., Year) or (Author1 & Author2, Year).

2. EVIDENCE ONLY: Only use information explicitly stated in the provided evidence chunks. Do NOT use any external knowledge or make inferences beyond what is directly stated.

3. MISSING INFORMATION: If the evidence does not address the question or lacks sufficient information, explicitly state: "This topic is not adequately addressed in the provided literature" or "The evidence does not contain information about X."

4. ACADEMIC TONE: Write in a formal, academic style suitable for a literature review. Use precise technical terminology.

5. SYNTHESIS: When multiple papers discuss the same topic, synthesize the information and cite all relevant sources.

6. CONTRADICTIONS: If papers contradict each other, acknowledge the disagreement and cite both perspectives.

7. SECTION CONTEXT: Pay attention to which section of a paper the evidence comes from (e.g., Introduction, Methods, Results). This provides context about the nature of the claim.

8. NO HALLUCINATION: 
   - Never invent citations, paper titles, paper details, journal names, or findings.
   - Use ONLY the exact paper titles provided in the evidence headers.
   - Do NOT create reference sections with full citations unless explicitly provided in the evidence.
   - If you need to reference a paper, use only the citation format (Author et al., Year) provided in the evidence.
   - If you're unsure, state that the evidence is unclear.
"""


def format_page_range(page_start: int = None, page_end: int = None) -> str:
    """
    Format page numbers for citation.
    
    Args:
        page_start: Starting page number
        page_end: Ending page number
    
    Returns:
        Formatted page range string (e.g., "p3-5" or "p3" or "")
    """
    if page_start is None:
        return ""
    if page_end is None or page_start == page_end:
        return f"p{page_start}"
    return f"p{page_start}-{page_end}"


def format_evidence_chunk(chunk: RetrievedChunk, index: int) -> str:
    """
    Format a single evidence chunk with citation, title, and metadata.
    
    Args:
        chunk: The retrieved chunk to format
        index: The evidence number (1-indexed)
    
    Returns:
        Formatted evidence string
    """
    page_info = format_page_range(chunk.page_start, chunk.page_end)
    
    # Build the header with citation, title, and metadata
    header_parts = [chunk.citation_pointer]
    if chunk.title:
        header_parts.append(f"Title: {chunk.title}")
    if chunk.section_name:
        header_parts.append(f"Section: {chunk.section_name}")
    if page_info:
        header_parts.append(page_info)
    
    header = " | ".join(header_parts)
    
    # Format the complete evidence entry
    evidence = f"Evidence {index} [{header}]:\n{chunk.text}"
    
    return evidence


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """
    Build a complete prompt for the LLM with system instruction, evidence, and question.
    
    Args:
        question: The user's question
        chunks: List of retrieved chunks to use as evidence
    
    Returns:
        Complete formatted prompt string
    """
    # Start with system instruction
    prompt_parts = [SYSTEM_INSTRUCTION]
    
    # Add evidence section
    if chunks:
        prompt_parts.append("\n" + "="*80)
        prompt_parts.append("EVIDENCE FROM LITERATURE:")
        prompt_parts.append("="*80 + "\n")
        
        for i, chunk in enumerate(chunks, 1):
            evidence = format_evidence_chunk(chunk, i)
            prompt_parts.append(evidence)
            prompt_parts.append("")  # Empty line between evidence chunks
        
        # Add a reference list at the end to prevent title hallucination
        prompt_parts.append("="*80)
        prompt_parts.append("REFERENCE LIST (Use these exact titles when referencing papers):")
        prompt_parts.append("="*80)
        seen_papers = {}
        for chunk in chunks:
            if chunk.paper_id not in seen_papers:
                seen_papers[chunk.paper_id] = {
                    "citation": chunk.citation_pointer,
                    "title": chunk.title
                }
        
        for paper_id, info in seen_papers.items():
            prompt_parts.append(f"- {info['citation']}: {info['title']}")
        prompt_parts.append("")
    else:
        prompt_parts.append("\n[No relevant evidence found in the literature.]")
    
    # Add the question
    prompt_parts.append("="*80)
    prompt_parts.append("USER QUESTION:")
    prompt_parts.append("="*80)
    prompt_parts.append(f"\n{question}\n")
    
    # Add instruction for answer
    prompt_parts.append("="*80)
    prompt_parts.append("YOUR ANSWER (with proper citations):")
    prompt_parts.append("="*80)
    prompt_parts.append("\nIMPORTANT: Use only the citation format (Author et al., Year) in your answer. Do NOT create full reference sections with journal names, volumes, or other details unless they are explicitly provided in the evidence above.\n")
    
    return "\n".join(prompt_parts)


def build_messages(question: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    """
    Build messages format for chat-based LLMs (like OpenAI's Chat API).
    
    Args:
        question: The user's question
        chunks: List of retrieved chunks to use as evidence
    
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    # System message
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]
    
    # Evidence in user message
    evidence_parts = []
    
    if chunks:
        evidence_parts.append("="*80)
        evidence_parts.append("EVIDENCE FROM LITERATURE:")
        evidence_parts.append("="*80 + "\n")
        
        for i, chunk in enumerate(chunks, 1):
            evidence = format_evidence_chunk(chunk, i)
            evidence_parts.append(evidence)
            evidence_parts.append("")
        
        # Add a reference list at the end to prevent title hallucination
        evidence_parts.append("="*80)
        evidence_parts.append("REFERENCE LIST (Use these exact titles when referencing papers):")
        evidence_parts.append("="*80)
        seen_papers = {}
        for chunk in chunks:
            if chunk.paper_id not in seen_papers:
                seen_papers[chunk.paper_id] = {
                    "citation": chunk.citation_pointer,
                    "title": chunk.title
                }
        
        for paper_id, info in seen_papers.items():
            evidence_parts.append(f"- {info['citation']}: {info['title']}")
        evidence_parts.append("")
    else:
        evidence_parts.append("[No relevant evidence found in the literature.]\n")
    
    # Question
    evidence_parts.append("="*80)
    evidence_parts.append("USER QUESTION:")
    evidence_parts.append("="*80)
    evidence_parts.append(f"\n{question}\n")
    
    evidence_parts.append("Please provide a comprehensive answer based on the evidence above, with proper citations.")
    evidence_parts.append("IMPORTANT: Use only the citation format (Author et al., Year) in your answer. Do NOT create full reference sections with journal names, volumes, or other details unless they are explicitly provided in the evidence above.")
    
    messages.append({
        "role": "user",
        "content": "\n".join(evidence_parts)
    })
    
    return messages


def extract_sources_from_chunks(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """
    Extract source information from chunks for the API response.
    
    Args:
        chunks: List of retrieved chunks
    
    Returns:
        List of source dictionaries
    """
    sources = []
    seen_papers = set()
    
    for chunk in chunks:
        # Avoid duplicate papers in sources
        if chunk.paper_id in seen_papers:
            continue
        
        seen_papers.add(chunk.paper_id)
        
        pages = []
        if chunk.page_start is not None:
            pages.append(chunk.page_start)
            if chunk.page_end is not None and chunk.page_end != chunk.page_start:
                pages.append(chunk.page_end)
        
        source = {
            "paper_id": chunk.paper_id,
            "citation": chunk.citation_pointer,
            "section": chunk.section_name,
            "pages": pages,
            "title": chunk.title
        }
        sources.append(source)
    
    return sources


if __name__ == "__main__":
    # Example usage
    from models import RetrievedChunk
    
    # Mock chunks for demonstration
    mock_chunks = [
        RetrievedChunk(
            text="Emotion recognition from gait patterns has shown promising results with accuracy rates exceeding 80% in controlled environments. The walking speed and stride length are particularly discriminative features for detecting emotional states.",
            title="Affective Computing and Emotion Recognition",
            citation_pointer="Narayanan et al., 2020",
            section_name="I. Introduction",
            page_start=1,
            page_end=1,
            paper_id="paper_001",
            year=2020,
            score=0.89,
            modality_tags=["gait"]
        ),
        RetrievedChunk(
            text="Our proxemic fusion framework combines spatial proximity data with emotional cues to predict pedestrian navigation behavior in crowded scenarios. Results show a 15% improvement over baseline methods.",
            title="Proxemic Fusion for Emotion-Aware Navigation",
            citation_pointer="Randhavane et al., 2019",
            section_name="III. Methodology",
            page_start=3,
            page_end=4,
            paper_id="paper_002",
            year=2019,
            score=0.85,
            modality_tags=["gait", "spatial"]
        )
    ]
    
    question = "How does emotion recognition from gait affect navigation behavior prediction?"
    
    print("=" * 80)
    print("SINGLE-STRING PROMPT FORMAT:")
    print("=" * 80)
    prompt = build_prompt(question, mock_chunks)
    print(prompt)
    
    print("\n" + "=" * 80)
    print("CHAT MESSAGES FORMAT:")
    print("=" * 80)
    messages = build_messages(question, mock_chunks)
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'])
    
    print("\n" + "=" * 80)
    print("EXTRACTED SOURCES:")
    print("=" * 80)
    sources = extract_sources_from_chunks(mock_chunks)
    for src in sources:
        print(f"- {src['citation']}: {src['title']} (Section: {src['section']}, Pages: {src['pages']})")


