"""
Module d'extraction multimodale utilisant Claude 3.7 Sonnet.
Extrait le contenu des images stockées dans MinIO.
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv
from anthropic import Anthropic
import requests
import time

# Import du gestionnaire MinIO
from .minio_manager import MinIOManager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalExtractor:
    """Extrait le contenu des images de pages PDF avec Claude."""
    
    def __init__(self):
        """Initialise le client Claude et MinIO."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-7-sonnet-latest"
        self.minio_manager = MinIOManager()
        
    def encode_image_from_url(self, image_url: str) -> str:
        """
        Télécharge et encode une image depuis MinIO.
        
        Args:
            image_url: URL MinIO de l'image
            
        Returns:
            Image encodée en base64
        """
        try:
            # Télécharger l'image depuis MinIO
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Encoder en base64
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur téléchargement image {image_url}: {e}")
            raise
    
    def encode_image(self, image_path: str) -> str:
        """Encode une image locale en base64 (garde pour compatibilité)."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_metadata_from_filename(self, filename: str) -> Dict:
        """
        Utilise Claude 3.7 Sonnet pour extraire intelligemment les métadonnées
        depuis n'importe quel format de nom de fichier PDF.
        """
        # Nettoyer le nom pour n'avoir que le nom du fichier sans le chemin
        clean_filename = Path(filename).name
        
        prompt = f"""Analyse ce nom de fichier PDF et extrais les métadonnées suivantes.
        
        Nom du fichier : {clean_filename}
        
        Extrais et retourne UNIQUEMENT un JSON avec ces champs :
        - date_publication : date au format YYYY-MM-DD si présente dans le nom
        - annee : année concernée par le contenu du document
        - periode : période couverte (exemples: "S1 2025", "T3 2024", "Exercice 2023", "Juin 2022", "Q2 2024")
        - type_document : type de rapport (exemples: "états financiers", "rapport d'activités", "synthèse", "résultats consolidés", "rapport de gestion")
        - organisation : nom de l'organisation (généralement "Sonatel" ou autre entreprise mentionnée)
        
        Si une information n'est pas disponible ou pas claire, mets null.
        
        Retourne UNIQUEMENT le JSON, sans aucune explication avant ou après :"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parser le JSON de la réponse
            metadata = json.loads(response.content[0].text.strip())
            # Ajouter le nom de fichier original
            metadata["filename"] = clean_filename
            logger.info(f"Métadonnées extraites pour {clean_filename}: {metadata}")
            return metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON pour {clean_filename}: {e}")
            return {
                "filename": clean_filename,
                "date_publication": None,
                "annee": None,
                "periode": None,
                "type_document": None,
                "organisation": None,
                "error": "Extraction des métadonnées échouée"
            }
        except Exception as e:
            logger.error(f"Erreur extraction métadonnées pour {clean_filename}: {e}")
            return {
                "filename": clean_filename,
                "error": str(e)
            }
    
    def extract_page_content(self, image_path: str, page_num: int) -> Dict:
        """
        Extrait le contenu d'une page depuis un fichier local (compatibilité).
        """
        try:
            # Encoder l'image
            base64_image = self.encode_image(image_path)
            
            # Prompt optimisé pour documents financiers
            prompt = """Extrais le contenu informationnel de cette page de document financier.

INSTRUCTIONS:
1. IGNORE les photos de fond et éléments purement décoratifs
2. EXTRAIS dans l'ordre d'apparition :
   - Tous les titres et sous-titres
   - Le texte complet des paragraphes
   - Les tableaux avec structure complète
   - Les graphiques avec description concise des données
   - Les schémas avec description factuelle brève

FORMAT pour chaque élément:
- TEXTE: Reproduis tel quel
- TABLEAUX: Format markdown avec toutes les colonnes et valeurs
- GRAPHIQUES: "[GRAPHIQUE: Type (ex: courbe, barres)] - Titre - Axes: X (libellé), Y (libellé) - Données clés visibles"
- SCHÉMAS: "[SCHÉMA] - Description factuelle en une ligne"

NE PAS:
- Interpréter ou analyser les données
- Décrire l'apparence visuelle générale
- Ajouter de contexte non présent

Extrais maintenant le contenu:"""

            # Appel à Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            return {
                "page_number": page_num,
                "image_path": image_path,
                "content": response.content[0].text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction page {page_num}: {e}")
            return {
                "page_number": page_num,
                "image_path": image_path,
                "content": "",
                "status": "error",
                "error": str(e)
            }
    
    def extract_page_content_from_minio(self, minio_url: str, page_num: int) -> Dict:
        """
        Extrait le contenu d'une page depuis MinIO.
        
        Args:
            minio_url: URL MinIO de l'image
            page_num: Numéro de la page
            
        Returns:
            Contenu extrait
        """
        try:
            # Télécharger et encoder l'image depuis MinIO
            base64_image = self.encode_image_from_url(minio_url)
            
            # Même prompt qu'avant
            prompt = """Extrais le contenu informationnel de cette page de document financier.

INSTRUCTIONS:
1. IGNORE les photos de fond et éléments purement décoratifs
2. EXTRAIS dans l'ordre d'apparition :
   - Tous les titres et sous-titres
   - Le texte complet des paragraphes
   - Les tableaux avec structure complète
   - Les graphiques avec description concise des données
   - Les schémas avec description factuelle brève

FORMAT pour chaque élément:
- TEXTE: Reproduis tel quel
- TABLEAUX: Format markdown avec toutes les colonnes et valeurs
- GRAPHIQUES: "[GRAPHIQUE: Type (ex: courbe, barres)] - Titre - Axes: X (libellé), Y (libellé) - Données clés visibles"
- SCHÉMAS: "[SCHÉMA] - Description factuelle en une ligne"

NE PAS:
- Interpréter ou analyser les données
- Décrire l'apparence visuelle générale
- Ajouter de contexte non présent

Extrais maintenant le contenu:"""

            # Appel à Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            return {
                "page_number": page_num,
                "minio_url": minio_url,
                "content": response.content[0].text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction page {page_num}: {e}")
            return {
                "page_number": page_num,
                "minio_url": minio_url,
                "content": "",
                "status": "error",
                "error": str(e)
            }
    
    def extract_all_pages_from_minio(
        self, 
        pages_info: List[Dict],
        output_dir: Path = None
    ) -> Dict:
        """
        Extrait le contenu de toutes les pages depuis MinIO.
        
        Args:
            pages_info: Liste des infos de pages (depuis pdf_to_images)
            output_dir: Dossier de sortie optionnel pour sauvegarder les extractions
            
        Returns:
            Dict contenant toutes les extractions
        """
        logger.info(f"Extraction de {len(pages_info)} pages depuis MinIO...")
        
        extractions = []
        
        for page_info in pages_info:
            page_num = page_info['page_number']
            minio_url = page_info['minio_url']
            
            logger.info(f"Extraction page {page_num}/{len(pages_info)}...")
            
            result = self.extract_page_content_from_minio(minio_url, page_num)
            extractions.append(result)
            
            # Sauvegarder si demandé
            if output_dir and result["status"] == "success":
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"page_{page_num:03d}.txt"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"=== PAGE {page_num} ===\n")
                    f.write(f"MinIO URL: {minio_url}\n")
                    f.write("="*50 + "\n\n")
                    f.write(result["content"])
            
            # Pause pour éviter rate limiting
            time.sleep(0.5)
        
        successful = sum(1 for e in extractions if e.get("status") == "success")
        logger.info(f"Extraction terminée: {successful}/{len(pages_info)} pages extraites")
        
        return {
            "extractions": extractions,
            "total_pages": len(extractions),
            "successful_pages": successful
        }
    
    # Garder l'ancienne méthode pour compatibilité
    def extract_all_pages(self, image_folder: str, output_dir: str = None, pdf_filename: str = None) -> Dict:
        """
        Méthode originale pour extraire depuis des images locales (compatibilité).
        """
        image_folder = Path(image_folder)
        image_files = sorted(image_folder.glob("*.png"))
        
        if not image_files:
            logger.warning(f"Aucune image trouvée dans {image_folder}")
            return {"extractions": [], "metadata": {}}
        
        doc_metadata = {}
        if pdf_filename:
            logger.info(f"Extraction des métadonnées depuis: {pdf_filename}")
            doc_metadata = self.extract_metadata_from_filename(pdf_filename)
        
        logger.info(f"Extraction de {len(image_files)} pages...")
        
        extractions = []
        for idx, image_file in enumerate(image_files, 1):
            logger.info(f"Extraction page {idx}/{len(image_files)}: {image_file.name}")
            result = self.extract_page_content(str(image_file), idx)
            
            result["document_metadata"] = doc_metadata
            extractions.append(result)
            
            if output_dir and result["status"] == "success":
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                output_file = output_path / f"page_{idx:03d}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"=== PAGE {idx} ===\n")
                    f.write(f"Source: {image_file.name}\n")
                    f.write("="*50 + "\n\n")
                    f.write(result["content"])
                    
        logger.info(f"Extraction terminée: {len(extractions)} pages traitées")
        
        return {
            "document_metadata": doc_metadata,
            "extractions": extractions,
            "total_pages": len(extractions),
            "successful_pages": sum(1 for e in extractions if e.get("status") == "success")
        }