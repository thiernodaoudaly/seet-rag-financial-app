"""
Module pour générer des embeddings à partir des chunks.
Utilise BGE-M3 pour de meilleures performances sur documents financiers multilingues.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Génère des embeddings pour les chunks de documents."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialise le générateur d'embeddings.
        
        Args:
            model_name: BGE-M3 - modèle multilingue optimisé pour le RAG
        """
        # Le modèle sera mis en cache dans ~/.cache/huggingface après le premier téléchargement
        cache_dir = Path.home() / ".cache" / "huggingface"
        logger.info(f"Cache des modèles: {cache_dir}")
        
        logger.info(f"Chargement du modèle d'embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Modèle chargé. Dimension des embeddings: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Génère des embeddings pour une liste de chunks.
        Compatible avec le format du pipeline MinIO.
        
        Args:
            chunks: Liste des chunks avec leur texte dans 'content'
            
        Returns:
            Liste des chunks enrichis avec leurs embeddings
        """
        if not chunks:
            logger.warning("Aucun chunk à traiter")
            return []
        
        # Extraire les textes - MODIFICATION ICI
        # Support pour les deux formats possibles
        texts = []
        for chunk in chunks:
            if 'content' in chunk:
                texts.append(chunk['content'])  # Format pipeline MinIO
            elif 'chunk_text' in chunk:
                texts.append(chunk['chunk_text'])  # Format ancien
            else:
                logger.error(f"Chunk sans contenu trouvé: {chunk.keys()}")
                texts.append("")
        
        logger.info(f"Génération des embeddings pour {len(texts)} chunks...")
        
        # Générer les embeddings en batch
        # BGE recommande de normaliser les embeddings
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=True  # Important pour BGE
        )
        
        # Ajouter les embeddings aux chunks existants
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            # Créer un nouveau dict avec la structure attendue par OpenSearch
            chunk_with_embedding = {
                "content": chunk.get('content') or chunk.get('chunk_text', ''),
                "embedding": embedding.tolist(),
                "metadata": chunk.get('metadata', {})
            }
            
            # Ajouter les infos du modèle dans les métadonnées
            chunk_with_embedding['metadata']['embedding_model'] = self.model_name
            chunk_with_embedding['metadata']['embedding_dim'] = self.embedding_dim
            
            chunks_with_embeddings.append(chunk_with_embedding)
        
        logger.info(f"{len(chunks_with_embeddings)} embeddings générés avec {self.model_name}")
        
        return chunks_with_embeddings
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict], output_path: Path):
        """
        Sauvegarde les chunks avec leurs embeddings.
        
        Args:
            chunks_with_embeddings: Chunks enrichis avec embeddings
            output_path: Chemin de sauvegarde
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Créer un document avec métadonnées
        output_data = {
            "model_used": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_chunks": len(chunks_with_embeddings),
            "chunks": chunks_with_embeddings
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Embeddings sauvegardés dans: {output_path}")
        
        # Afficher la taille du fichier
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Taille du fichier: {file_size_mb:.2f} MB")
    
    def load_embeddings(self, embeddings_path: Path) -> Dict:
        """
        Charge des embeddings précédemment sauvegardés.
        
        Args:
            embeddings_path: Chemin vers le fichier d'embeddings
            
        Returns:
            Dictionnaire contenant les embeddings et métadonnées
        """
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Embeddings chargés: {data['total_chunks']} chunks")
        logger.info(f"Modèle utilisé: {data['model_used']}")
        logger.info(f"Dimension: {data['embedding_dimension']}")
        
        return data