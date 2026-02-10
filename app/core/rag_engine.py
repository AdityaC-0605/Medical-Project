"""
RAG Engine for Medical Knowledge Retrieval
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Medical document."""
    id: str
    content: str
    domain: str


class SimpleRAG:
    """Lightweight RAG using keyword matching."""
    
    def __init__(self):
        self.documents = self._load_documents()
        logger.info(f"RAG loaded with {len(self.documents)} documents")
    
    def _load_documents(self) -> List[Document]:
        """Load medical knowledge base."""
        docs = [
            Document(
                id="cardio_guidelines",
                domain="cardiology",
                content="""
Cardiovascular Risk Assessment:
- LDL Cholesterol: Optimal <100, Borderline 130-159, High 160-189, Very High ≥190 mg/dL
- HDL Cholesterol: Low <40 (men), <50 (women), Optimal ≥60 mg/dL
- Triglycerides: Normal <150, Borderline 150-199, High 200-499, Very High ≥500 mg/dL
- Total Cholesterol: Desirable <200, Borderline 200-239, High ≥240 mg/dL
- Risk factors: Age, sex, smoking, BP, diabetes, family history
- Management: Lifestyle first, statins for high risk (LDL >190 or ASCVD risk >7.5%)
"""
            ),
            Document(
                id="ct_coronary_guidelines",
                domain="radiology",
                content="""
CT Coronary Angiography Interpretation:
- Stenosis Severity: Minimal <25%, Mild 25-49%, Moderate 50-69%, Severe 70-89%, Critical ≥90%
- Hemodynamic Significance: >70% generally significant, 50-70% may be significant
- Plaque Types: Calcified (stable, low rupture risk), Soft (vulnerable), Mixed
- Management: <50% medical therapy, 50-70% functional assessment, >70% consider revascularization
- Clinical correlation essential for decision making
"""
            ),
            Document(
                id="cancer_pathology",
                domain="pathology",
                content="""
Cancer Grading and Markers:
- Histologic Grade: Grade 1 (well differentiated), Grade 2 (moderate), Grade 3 (poor)
- Breast Cancer Markers: ER+ (hormone therapy), PR+ (hormone therapy), HER2+ (targeted therapy)
- Prognosis factors: Tumor size, grade, nodal status, receptor status, Ki-67
- Treatment planning requires multidisciplinary team approach
"""
            ),
            Document(
                id="lipid_management",
                domain="cardiology",
                content="""
Lipid Management Guidelines:
- High-intensity statins: LDL reduction ≥50% (Atorvastatin 40-80mg, Rosuvastatin 20-40mg)
- Moderate-intensity statins: LDL reduction 30-49%
- Add ezetimibe if LDL goal not met on maximally tolerated statin
- Consider PCSK9 inhibitors for very high risk or familial hypercholesterolemia
- Lifestyle: Mediterranean diet, exercise 150min/week, weight loss, smoking cessation
- Follow-up: Lipid panel 4-12 weeks after initiation, then every 3-12 months
"""
            ),
            Document(
                id="cad_management",
                domain="cardiology",
                content="""
Coronary Artery Disease Management:
- Medical therapy: Antiplatelet (Aspirin), Statin (high-intensity), Beta-blocker, ACE-I
- Symptom control: Nitrates, Calcium channel blockers for angina
- Risk factor modification: BP <130/80, LDL <70 for established CAD, HbA1c <7%
- Revascularization: Consider for significant stenosis with symptoms or high-risk features
- Cardiac rehabilitation recommended for all patients post-MI or post-revascularization
"""
            )
        ]
        return docs
    
    def retrieve(self, query: str, domain: Optional[str] = None, top_k: int = 3) -> str:
        """
        Retrieve relevant medical knowledge.
        
        Args:
            query: Search query
            domain: Filter by domain (cardiology, radiology, pathology)
            top_k: Number of documents to retrieve
        
        Returns:
            Combined context string
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score documents
        scored_docs = []
        for doc in self.documents:
            # Domain filter
            if domain and doc.domain != domain:
                continue
            
            # Simple keyword scoring
            score = 0
            doc_lower = doc.content.lower()
            for word in query_words:
                if len(word) > 3 and word in doc_lower:
                    score += 1
            
            # Domain match bonus
            if domain and doc.domain == domain:
                score += 5
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]
        
        # Combine context
        if not top_docs:
            # Return domain-specific or general knowledge
            if domain:
                context = "\n\n".join([d.content for d in self.documents if d.domain == domain])
            else:
                context = "\n\n".join([d.content for d in self.documents[:2]])
        else:
            context = "\n\n".join([d.content for d, _ in top_docs])
        
        logger.info(f"RAG: Retrieved {len(top_docs)} documents ({len(context)} chars)")
        return context
    
    def get_statistics(self) -> Dict:
        """Get RAG statistics."""
        domains = {}
        for doc in self.documents:
            domains[doc.domain] = domains.get(doc.domain, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "domains": domains
        }
