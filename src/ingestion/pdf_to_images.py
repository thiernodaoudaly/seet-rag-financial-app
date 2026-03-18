"""
Module pour convertir les pages PDF en images et les stocker dans MinIO.
"""

import fitz  # PyMuPDF
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict
import logging
import tempfile
import shutil
from datetime import datetime

# Import du gestionnaire MinIO
from .minio_manager import MinIOManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToImageConverter:
    """Convertit un PDF en images et les stocke dans MinIO."""
    
    def __init__(self, dpi: int = 150):
        """
        Initialise le convertisseur.
        
        Args:
            dpi: Résolution des images (150-200 DPI recommandé)
        """
        self.dpi = dpi
        self.zoom = dpi / 72.0
        self.minio_manager = MinIOManager()
        
    def convert_pdf_to_minio(
        self, 
        pdf_path: str, 
        doc_metadata: Dict
    ) -> List[Dict]:
        """
        Convertit un PDF en images et les upload vers MinIO.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            doc_metadata: Métadonnées extraites par Claude
            
        Returns:
            Liste de dictionnaires avec infos de chaque page
        """
        # Vérifier si le document existe déjà
        if self.minio_manager.check_document_exists(doc_metadata):
            logger.warning(f"Document déjà présent dans MinIO: {doc_metadata.get('filename')}")
            # Optionnel : supprimer l'ancien
            # self.minio_manager.delete_document(doc_metadata)
        
        # Créer un dossier temporaire
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Ouvrir le PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = pdf_document.page_count
            logger.info(f"PDF ouvert: {pdf_path} ({total_pages} pages)")
            
            pages_info = []
            
            # Convertir chaque page
            for page_num in range(total_pages):
                try:
                    # Charger la page
                    page = pdf_document[page_num]
                    
                    # Créer une matrice de transformation
                    matrix = fitz.Matrix(self.zoom, self.zoom)
                    
                    # Rendre la page en image
                    pix = page.get_pixmap(matrix=matrix)
                    
                    # Sauvegarder temporairement
                    temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num + 1:03d}.png")
                    pix.save(temp_image_path)
                    
                    # Upload vers MinIO avec la nouvelle structure
                    result = self.minio_manager.upload_document_page(
                        local_image_path=temp_image_path,
                        doc_metadata=doc_metadata,
                        page_num=page_num + 1,
                        total_pages=total_pages
                    )
                    
                    # Ajouter les dimensions
                    result["width"] = pix.width
                    result["height"] = pix.height
                    
                    pages_info.append(result)
                    
                    logger.info(f"Page {page_num + 1} uploadée: {result['minio_path']}")
                    
                    # Supprimer l'image temporaire immédiatement
                    os.remove(temp_image_path)
                    
                except Exception as e:
                    logger.error(f"Erreur page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
            # Log de la structure créée
            doc_id = self.minio_manager.generate_document_id(doc_metadata.get('filename'))
            logger.info(f"Document uploadé avec ID: {doc_id}")
            logger.info(f"Structure: {doc_metadata.get('organisation')}/{doc_metadata.get('annee')}/{doc_metadata.get('periode')}/{doc_id}/")
            logger.info(f"Total: {len(pages_info)} pages dans MinIO")
            
            return pages_info
            
        finally:
            # Nettoyer le dossier temporaire
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Dossier temporaire nettoyé")