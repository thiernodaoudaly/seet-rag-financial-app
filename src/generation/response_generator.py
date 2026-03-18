"""
Module de génération de réponses avec Claude.
Prend les chunks du retriever et génère une réponse contextualisée.
"""

import os
from typing import List, Dict, Optional, Tuple
from anthropic import Anthropic
import logging
from pathlib import Path
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Génère des réponses avec Claude basées sur le contexte RAG."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le générateur de réponses.
        
        Args:
            api_key: Clé API Anthropic (ou utilise ANTHROPIC_API_KEY env)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API Anthropic manquante")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-3-7-sonnet-latest"
        logger.info(f"Générateur initialisé avec {self.model}")
    
    def format_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Formate les chunks en contexte pour Claude.
        Compatible avec la structure V2 où les métadonnées sont dans 'metadata'.
        
        Args:
            chunks: Liste des chunks du retriever
            
        Returns:
            Tuple (contexte formaté, infos détaillées de chaque chunk)
        """
        context_parts = []
        chunks_info = []
        
        for i, chunk in enumerate(chunks, 1):
            # Dans V2, metadata contient toutes les infos au niveau racine
            metadata = chunk.get('metadata', {})
            
            # Récupérer les infos - d'abord depuis le chunk direct, puis metadata
            filename = chunk.get('filename') or metadata.get('filename', 'Document inconnu')
            page_num = chunk.get('page_number') or metadata.get('page_number', 0)
            section = chunk.get('section') or metadata.get('section', '')
            content_type = chunk.get('content_type') or metadata.get('content_type', '')
            periode = chunk.get('periode') or metadata.get('periode', '')
            annee = chunk.get('annee') or metadata.get('annee', '')
            minio_url = chunk.get('minio_url') or metadata.get('minio_url', '')
            type_document = chunk.get('type_document') or metadata.get('type_document', '')
            
            # Formater le contexte pour Claude
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Fichier: {filename}")
            context_parts.append(f"Page: {page_num}")
            
            if type_document:
                context_parts.append(f"Type document: {type_document}")
            if section:
                context_parts.append(f"Section: {section}")
            if content_type:
                context_parts.append(f"Type contenu: {content_type}")
            if periode:
                context_parts.append(f"Période: {periode}")
            if annee:
                context_parts.append(f"Année: {annee}")
            
            # Ajouter le contenu
            content = chunk.get('chunk_text', '')
            context_parts.append(f"Contenu:\n{content}")
            context_parts.append("-" * 50)
            
            # Sauvegarder les infos complètes
            chunks_info.append({
                'filename': filename,
                'page': page_num,
                'section': section,
                'content_type': content_type,
                'type_document': type_document,
                'periode': periode,
                'annee': annee,
                'minio_url': minio_url,
                'organisation': chunk.get('organisation') or metadata.get('organisation', '')
            })
        
        return "\n".join(context_parts), chunks_info
    
    def select_images_from_cited_pages(
        self, 
        chunks_info: List[Dict], 
        cited_pages: List[str],
        max_images: int = 2
    ) -> List[Dict]:
        """
        Sélectionne les images correspondant aux pages citées par Claude.
        
        Args:
            chunks_info: Informations des chunks utilisés
            cited_pages: Liste des numéros de pages citées par Claude
            max_images: Nombre maximum d'images à retourner
            
        Returns:
            Liste des images des pages citées
        """
        if not cited_pages:
            logger.info("Aucune page citée, pas d'images à sélectionner")
            return []
        
        # Convertir les pages citées en entiers
        cited_page_nums = []
        for page in cited_pages:
            try:
                cited_page_nums.append(int(page))
            except ValueError:
                logger.warning(f"Page invalide ignorée: {page}")
        
        logger.info(f"Pages citées par Claude: {cited_page_nums}")
        
        # Filtrer les chunks qui correspondent aux pages citées ET qui ont une URL MinIO
        matching_images = []
        
        for chunk_info in chunks_info:
            if not chunk_info.get('minio_url'):
                continue
            
            page_num = chunk_info.get('page')
            if page_num in cited_page_nums:
                # Score de priorité basé sur le type de contenu
                content_type = chunk_info.get('content_type', '').lower()
                priority_score = 1
                
                if any(word in content_type for word in ['tableau', 'table', 'graphique', 'chart', 'graph']):
                    priority_score = 3
                elif any(word in content_type for word in ['kpi', 'indicateur', 'chiffre']):
                    priority_score = 2
                
                matching_images.append({
                    'url': chunk_info['minio_url'],
                    'page': chunk_info['page'],
                    'type': chunk_info.get('content_type', ''),
                    'description': chunk_info.get('section', ''),
                    'filename': chunk_info['filename'],
                    'periode': chunk_info.get('periode', ''),
                    'annee': chunk_info.get('annee', ''),
                    'priority': priority_score
                })
        
        if not matching_images:
            logger.warning(f"Aucune image trouvée pour les pages citées {cited_page_nums}")
            return []
        
        # Trier par priorité (type de contenu) puis par numéro de page
        matching_images.sort(key=lambda x: (x['priority'], x['page']), reverse=True)
        
        # Limiter au nombre max d'images
        selected = matching_images[:max_images]
        logger.info(f"Images sélectionnées: Pages {[img['page'] for img in selected]}")
        
        return selected
    
    def generate_response(
        self,
        query: str,
        chunks: List[Dict],
        include_sources: bool = True,
        include_images: bool = False,
        max_tokens: int = 1500
    ) -> Dict:
        """
        Génère une réponse basée sur les chunks récupérés.
        
        Args:
            query: Question de l'utilisateur
            chunks: Chunks pertinents du retriever
            include_sources: Inclure les sources dans la réponse
            include_images: Inclure les URLs des images MinIO
            max_tokens: Limite de tokens pour la réponse
            
        Returns:
            Dictionnaire avec réponse et métadonnées
        """
        if not chunks:
            return {
                "response": "Je n'ai pas trouvé d'informations pertinentes dans les documents disponibles pour répondre à votre question.",
                "sources": [],
                "chunks_used": 0,
                "images": []
            }
        
        # Formater le contexte et récupérer les infos des chunks
        context, chunks_info = self.format_context(chunks)
        
        # Construire le prompt système adaptatif selon le contexte
        system_prompt = """Tu es un assistant expert en analyse de documents.
        
Règles importantes:
1. Base tes réponses UNIQUEMENT sur les informations fournies dans le contexte
2. Cite précisément les chiffres et données avec leurs unités
3. Si une information n'est pas dans le contexte, dis-le clairement
4. Structure ta réponse de manière claire
5. Utilise les données les plus récentes disponibles dans le contexte
6. Mentionne TOUJOURS de quel document/période proviennent les informations principales"""
        
        # Construire le prompt utilisateur
        user_prompt = f"""Contexte des documents:
{context}

Question: {query}

Réponds de manière précise et structurée en te basant uniquement sur le contexte fourni."""
        
        if include_sources:
            user_prompt += "\n\nIndique les pages sources entre crochets [Page X] quand tu cites des informations spécifiques."
        
        # Appel à Claude
        logger.info(f"Génération de réponse pour: {query[:50]}...")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            answer = response.content[0].text
            
            # Extraire les pages citées dans la réponse
            cited_pages = list(set(re.findall(r'\[Page (\d+)\]', answer)))
            logger.info(f"Pages extraites de la réponse: {cited_pages}")
            
            # Sélectionner les images basées sur les pages citées
            relevant_images = []
            if include_images:
                relevant_images = self.select_images_from_cited_pages(
                    chunks_info, 
                    cited_pages,
                    max_images=2
                )
            
            # Préparer les sources
            sources = []
            for chunk_info in chunks_info:
                sources.append({
                    "filename": chunk_info['filename'],
                    "page": chunk_info['page'],
                    "section": chunk_info['section'],
                    "type": chunk_info.get('content_type', ''),
                    "type_document": chunk_info.get('type_document', ''),
                    "periode": chunk_info.get('periode', ''),
                    "annee": chunk_info.get('annee', '')
                })
            
            return {
                "response": answer,
                "sources": sources,
                "chunks_used": len(chunks),
                "cited_pages": cited_pages,
                "images": relevant_images,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return {
                "response": f"Erreur lors de la génération de la réponse: {str(e)}",
                "sources": [],
                "chunks_used": 0,
                "images": [],
                "error": str(e)
            }