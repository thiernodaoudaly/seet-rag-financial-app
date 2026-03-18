"""
Module de chunking sémantique intelligent utilisant Claude 3.7 Sonnet.
Adaptatif pour tous types de documents financiers.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dotenv import load_dotenv
from anthropic import Anthropic
import hashlib
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """Découpe intelligente de documents en chunks sémantiques."""
    
    def __init__(self):
        """Initialise le chunker avec Claude."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-7-sonnet-latest"
        
    def clean_page_content(self, page_content: str) -> Tuple[str, int]:
        """
        Nettoie le contenu d'une page en enlevant les en-têtes.
        Retourne le contenu nettoyé et le numéro de page.
        """
        lines = page_content.split('\n')
        
        # Extraire le numéro de page depuis l'en-tête
        page_num = 0
        for line in lines[:5]:
            if "PAGE" in line:
                match = re.search(r'PAGE\s+(\d+)', line)
                if match:
                    page_num = int(match.group(1))
                    break
        
        # Trouver où commence le contenu réel
        content_start = 0
        for i, line in enumerate(lines):
            if line.startswith("="*50):
                content_start = i + 2
                break
        
        # Retourner le contenu nettoyé
        if content_start < len(lines):
            clean_content = '\n'.join(lines[content_start:]).strip()
        else:
            clean_content = page_content.strip()
            
        return clean_content, page_num
    
    def create_chunks(self, text: str, metadata: dict = None) -> List[Dict]:
        """
        Interface simple pour créer des chunks depuis du texte en mémoire.
        Compatible avec le pipeline MinIO.
        
        Args:
            text: Texte à découper
            metadata: Métadonnées à attacher à chaque chunk
        """
        if not metadata:
            metadata = {}
        
        page_num = metadata.get('page_number', 1)
        doc_type = metadata.get('type_document', 'document financier')
        
        # Utiliser la méthode existante chunk_with_claude
        chunks_data = self.chunk_with_claude(text, page_num, doc_type)
        
        # Enrichir chaque chunk avec toutes les métadonnées
        enriched_chunks = []
        for chunk in chunks_data:
            enriched_chunk = {
                "content": chunk['chunk_text'],  # Renommer pour compatibilité
                "metadata": {
                    **metadata,  # TOUTES les métadonnées incluant minio_url
                    "chunk_id": chunk['chunk_id'],
                    "chunk_index": chunk['chunk_index'],
                    "content_type": chunk['content_type'],
                    "section": chunk['section'],
                    "keywords": chunk['keywords'],
                    "has_numbers": chunk['has_numbers'],
                    "is_complete": chunk['is_complete']
                }
            }
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def chunk_with_claude(self, content: str, page_num: int, doc_type: str = None) -> List[Dict]:
        """
        Utilise Claude pour diviser intelligemment le contenu en chunks.
        """
        context = ""
        if doc_type:
            context = f"Type de document: {doc_type}\n"
        
        prompt = f"""Analyse ce contenu de document financier et divise-le en chunks sémantiques cohérents.

{context}
RÈGLES CRITIQUES DE CHUNKING:

1. TAILLE ET COHÉRENCE:
   - Taille minimale: 100 caractères (sauf pour les titres de sections isolés)
   - Taille idéale: 300-800 mots pour le texte narratif
   - Les petits éléments isolés (comme "avec vous, pour vous" ou numéros de page seuls) doivent être fusionnés avec le contenu adjacent
   - Un chunk doit être autonome et compréhensible seul

2. PRÉSERVATION DES STRUCTURES:
   - NE JAMAIS diviser un tableau - le garder entier
   - NE JAMAIS séparer un titre de son contenu immédiat
   - Les listes à puces complètes restent ensemble
   - Les graphiques [GRAPHIQUE:...] restent avec leur contexte

3. ÉVITER LES DUPLICATIONS:
   - Si tu détectes du contenu répété, ne créer qu'un seul chunk
   - Les en-têtes/pieds de page répétitifs doivent être ignorés ou groupés

4. REGROUPEMENT INTELLIGENT:
   - Fusionner les éléments trop courts avec le contenu pertinent adjacent
   - Un slogan seul doit être attaché au contenu principal de la page
   - Les numéros de page isolés ne sont PAS des chunks séparés

TYPES DE CONTENU (être TRÈS PRÉCIS et ne te limite pas seulement à cette liste spécifique c'est juste pour te guider):
- "page_couverture": Page de titre/couverture complète
- "sommaire": Table des matières
- "définitions": Lexique ou glossaire
- "résumé_exécutif": Synthèse ou points clés
- "faits_marquants": Événements importants par pays/région
- "kpi_financiers": Indicateurs clés avec données chiffrées
- "tableau_financier": Tableau complet de données
- "graphique_analyse": Graphique avec son analyse
- "compte_résultat": Compte de résultat
- etc.

Pour CHAQUE chunk identifié, créer un objet JSON:
{{
  "chunk_text": "Texte EXACT et COMPLET du chunk",
  "content_type": "type précis de contenu comme ce que je demande ci-dessus",
  "section": "Titre de la section principale",
  "keywords": ["3 à 7 mots-clés pertinents"],
  "has_numbers": true/false,
  "is_complete": true/false (le chunk est-il une unité complète d'information?)
}}

CONTENU À ANALYSER:
---
{content}
---

IMPORTANT: 
- Regroupe intelligemment les éléments courts
- Évite les chunks de moins de 100 caractères
- Détecte et élimine les duplications

Retourne UNIQUEMENT un array JSON valide des chunks:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parser la réponse JSON
            response_text = response.content[0].text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            chunks_data = json.loads(response_text)
            
            # Post-traitement : filtrer les chunks trop courts et vérifier la qualité
            enriched_chunks = []
            for idx, chunk in enumerate(chunks_data):
                chunk_text = chunk.get('chunk_text', '').strip()
                
                # Ignorer les chunks vides ou trop courts (sauf titres de section)
                if len(chunk_text) < 50 and chunk.get('content_type') not in ['page_couverture', 'sommaire']:
                    logger.warning(f"Chunk trop court ignoré sur page {page_num}: {chunk_text[:30]}...")
                    continue
                
                # Générer un ID unique
                chunk_id = self.generate_chunk_id(page_num, len(enriched_chunks), chunk_text)
                
                enriched_chunk = {
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "chunk_index": len(enriched_chunks),
                    "chunk_text": chunk_text,
                    "content_type": chunk.get('content_type', 'texte_narratif'),
                    "section": chunk.get('section', ''),
                    "keywords": chunk.get('keywords', []),
                    "has_numbers": chunk.get('has_numbers', False),
                    "is_complete": chunk.get('is_complete', True)
                }
                enriched_chunks.append(enriched_chunk)
            
            # Si aucun chunk valide, créer un chunk unique avec tout le contenu
            if not enriched_chunks and len(content.strip()) > 50:
                return self.create_fallback_chunk(content, page_num)
            
            return enriched_chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON page {page_num}: {e}")
            return self.create_fallback_chunk(content, page_num)
        except Exception as e:
            logger.error(f"Erreur chunking page {page_num}: {e}")
            return self.create_fallback_chunk(content, page_num)
    
    def create_fallback_chunk(self, content: str, page_num: int) -> List[Dict]:
        """
        Crée un chunk de fallback si l'analyse Claude échoue.
        """
        chunk_id = self.generate_chunk_id(page_num, 0, content)
        return [{
            "chunk_id": chunk_id,
            "page_number": page_num,
            "chunk_index": 0,
            "chunk_text": content,
            "content_type": "texte_narratif",
            "section": f"Page {page_num}",
            "keywords": [],
            "has_numbers": bool(re.search(r'\d+', content)),
            "is_complete": True
        }]
    
    def generate_chunk_id(self, page_num: int, chunk_index: int, text: str) -> str:
        """
        Génère un ID unique pour un chunk.
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"p{page_num:03d}_c{chunk_index:02d}_{text_hash}"
    
    def deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Élimine les chunks dupliqués en gardant la première occurrence.
        """
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk['chunk_text'].strip()
            # Utiliser un hash pour la comparaison (plus efficace)
            text_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_chunks.append(chunk)
            else:
                logger.info(f"Chunk dupliqué éliminé: {chunk['chunk_id']}")
        
        # Réindexer les chunks
        for idx, chunk in enumerate(unique_chunks):
            chunk['chunk_index'] = idx
            
        return unique_chunks
    
    def process_document(self, extraction_dir: Path, doc_metadata: Dict) -> Dict:
        """
        Traite un document complet extrait.
        """
        page_files = sorted(extraction_dir.glob("page_*.txt"))
        
        if not page_files:
            logger.error(f"Aucun fichier de page trouvé dans {extraction_dir}")
            return {"chunks": [], "metadata": doc_metadata}
        
        logger.info(f"Traitement de {len(page_files)} pages pour chunking...")
        
        all_chunks = []
        stats = {
            "total_pages": len(page_files),
            "total_chunks": 0,
            "chunks_by_type": {},
            "duplicates_removed": 0,
            "short_chunks_merged": 0
        }
        
        doc_type = doc_metadata.get('type_document', 'document financier')
        
        for page_file in page_files:
            logger.info(f"Chunking {page_file.name}...")
            
            # Lire et nettoyer le contenu
            content = page_file.read_text(encoding='utf-8')
            clean_content, page_num = self.clean_page_content(content)
            
            # Si la page est vide ou trop courte, passer
            if len(clean_content.strip()) < 50:
                logger.warning(f"Page {page_num} trop courte, ignorée")
                stats['short_chunks_merged'] += 1
                continue
            
            # Chunker le contenu
            page_chunks = self.chunk_with_claude(clean_content, page_num, doc_type)
            
            # Ajouter les métadonnées du document à chaque chunk
            for chunk in page_chunks:
                # Hériter TOUTES les métadonnées du document
                chunk.update({
                    "filename": doc_metadata.get('filename'),
                    "date_publication": doc_metadata.get('date_publication'),
                    "annee": doc_metadata.get('annee'),
                    "periode": doc_metadata.get('periode'),
                    "type_document": doc_metadata.get('type_document'),
                    "organisation": doc_metadata.get('organisation')
                })
                
                all_chunks.append(chunk)
            
            # Pause pour éviter rate limiting
            time.sleep(0.5)
        
        # Déduplication finale
        initial_count = len(all_chunks)
        all_chunks = self.deduplicate_chunks(all_chunks)
        stats['duplicates_removed'] = initial_count - len(all_chunks)
        
        # Calculer les statistiques finales
        stats['total_chunks'] = len(all_chunks)
        for chunk in all_chunks:
            content_type = chunk.get('content_type', 'unknown')
            stats['chunks_by_type'][content_type] = stats['chunks_by_type'].get(content_type, 0) + 1
        
        logger.info(f"Chunking terminé: {stats['total_chunks']} chunks créés")
        logger.info(f"Duplicatas supprimés: {stats['duplicates_removed']}")
        logger.info(f"Distribution par type: {stats['chunks_by_type']}")
        
        return {
            "document_metadata": doc_metadata,
            "chunks": all_chunks,
            "statistics": stats,
            "extraction_dir": str(extraction_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_chunks(self, chunks_data: Dict, output_dir: Path) -> Path:
        """
        Sauvegarde les chunks dans un fichier JSON.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = chunks_data['document_metadata'].get('filename', 'unknown')
        base_name = Path(filename).stem
        output_file = output_dir / f"{base_name}_chunks.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chunks sauvegardés dans: {output_file}")
        return output_file