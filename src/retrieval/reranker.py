"""
Reranking des chunks avec Claude pour améliorer la pertinence.
"""

import os
from typing import List, Dict
from anthropic import Anthropic
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeReranker:
    """Rerank les chunks avec Claude pour sélectionner les plus pertinents."""
    
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-haiku-20240307"
        logger.info("Reranker initialisé avec Claude Haiku")
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank les chunks selon leur pertinence pour la query.
        
        Args:
            query: Question de l'utilisateur
            chunks: Liste des chunks récupérés
            top_k: Nombre de chunks à retourner
        """
        if len(chunks) <= top_k:
            logger.info(f"Pas besoin de reranking: {len(chunks)} chunks <= {top_k}")
            return chunks
        
        logger.info(f"Reranking: {len(chunks)} chunks → top {top_k}")
        
        # Créer un résumé de chaque chunk
        chunks_summary = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            
            page = chunk.get('page_number') or metadata.get('page_number', 0)
            section = chunk.get('section') or metadata.get('section', 'N/A')
            content_type = chunk.get('content_type') or metadata.get('content_type', 'N/A')
            text = chunk.get('chunk_text', '')[:800]
            
            chunks_summary.append({
                "id": i,
                "page": page,
                "section": section,
                "type": content_type,
                "preview": text
            })
        
        # Construire le prompt
        prompt = self._build_prompt(query, chunks_summary, top_k)
        
        # Log de la taille du prompt pour monitoring
        prompt_length = len(prompt)
        logger.info(f"Taille du prompt de reranking: {prompt_length} caractères (~{prompt_length//4} tokens)")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,  
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parser la réponse
            response_text = response.content[0].text.strip()
            
            # Nettoyer la réponse si elle contient des backticks markdown
            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            selected_ids = result["selected_ids"][:top_k]
            
            # Retourner les chunks sélectionnés dans l'ordre
            reranked = [chunks[id] for id in selected_ids if id < len(chunks)]
            
            logger.info(f"Reranking terminé: {len(reranked)} chunks sélectionnés")
            logger.info(f"IDs sélectionnés: {selected_ids}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Erreur reranking: {e}")
            logger.warning("Fallback: retour des premiers chunks")
            return chunks[:top_k]
    
    def _build_prompt(self, query: str, chunks_summary: List[Dict], top_k: int) -> str:
        """Construit le prompt pour Claude."""
        chunks_text = self._format_chunks(chunks_summary)
        
        prompt = f"""Tu dois sélectionner les {top_k} chunks les PLUS pertinents pour répondre à cette question.

Question: {query}

Chunks disponibles:
{chunks_text}

Critères de sélection:
1. Pertinence directe avec la question (le plus important)
2. Diversité des informations (éviter les doublons)
3. Complétude (préférer les chunks avec données concrètes)
4. Éviter les pages de couverture ou sommaires sauf si pertinents

Retourne UNIQUEMENT un JSON avec les IDs des {top_k} meilleurs chunks, du PLUS pertinent au moins pertinent.
Format exact attendu (sans backticks):
{{"selected_ids": [3, 7, 1, 9, 4]}}"""
        
        return prompt
    
    def _format_chunks(self, chunks_summary: List[Dict]) -> str:
        """Formate les chunks pour le prompt."""
        lines = []
        for c in chunks_summary:
            lines.append(f"[ID {c['id']}] Page {c['page']} | Section: {c['section']} | Type: {c['type']}")
            lines.append(f"  Contenu: {c['preview']}...")  
            lines.append("")
        return "\n".join(lines)