"""
Module de gestion du stockage MinIO pour les images PDF.
"""

from minio import Minio
from minio.error import S3Error
import logging
from pathlib import Path
import io
from typing import List, Optional, Tuple, Dict
import hashlib
from datetime import datetime
import unicodedata
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinIOManager:
    """Gère le stockage et la récupération des images dans MinIO."""
    
    def __init__(
        self, 
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False
    ):
        """
        Initialise la connexion à MinIO.
        
        Args:
            endpoint: Adresse du serveur MinIO
            access_key: Clé d'accès
            secret_key: Clé secrète
            secure: Utiliser HTTPS (False pour local)
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        # Nom du bucket pour les documents
        self.bucket_name = "rag-documents"
        
        # Créer le bucket s'il n'existe pas
        self._ensure_bucket_exists()
        
        logger.info(f"Connexion MinIO établie: {endpoint}")
        logger.info(f"Bucket actif: {self.bucket_name}")
    
    def _ensure_bucket_exists(self):
        """Crée le bucket s'il n'existe pas."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' créé")
            else:
                logger.info(f"Bucket '{self.bucket_name}' existe déjà")
        except S3Error as e:
            logger.error(f"Erreur création bucket: {e}")
            raise
    
    def remove_accents(self, text: str) -> str:
        """
        Enlève les accents d'un texte pour compatibilité ASCII.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte sans accents
        """
        if not text:
            return text
        # Normaliser et enlever les accents
        nfd = unicodedata.normalize('NFD', str(text))
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    def generate_document_id(self, filename: str) -> str:
        """
        Utilise simplement le nom du fichier sans extension.
        
        Args:
            filename: Nom du fichier PDF
            
        Returns:
            Nom du fichier sans extension
        """
        # Simplement retourner le nom sans l'extension .pdf
        return Path(filename).stem
    
    def build_object_path(self, doc_metadata: Dict, page_num: int) -> str:
        """
        Construit le chemin complet de l'objet dans MinIO.
        
        Structure : organisation/année/période/nom_document/page_xxx.png
        
        Args:
            doc_metadata: Métadonnées du document
            page_num: Numéro de la page
            
        Returns:
            Chemin complet dans MinIO
        """
        # Extraire et nettoyer les métadonnées pour le chemin
        organisation = self.remove_accents(
            doc_metadata.get('organisation', 'unknown')
        ).lower().replace(' ', '_')
        
        annee = str(doc_metadata.get('annee', 'unknown'))
        
        periode = self.remove_accents(
            doc_metadata.get('periode', 'unknown')
        ).lower().replace(' ', '_')
        
        filename = doc_metadata.get('filename', 'unknown')
        
        # Utiliser le nom complet du fichier sans extension
        doc_id = self.generate_document_id(filename)
        
        # Construire le chemin : organisation/année/période/nom_document/page_xxx.png
        object_path = f"{organisation}/{annee}/{periode}/{doc_id}/page_{page_num:03d}.png"
        
        return object_path
    
    def upload_image(
        self, 
        local_path: str, 
        object_name: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload une image vers MinIO.
        
        Args:
            local_path: Chemin local de l'image
            object_name: Nom de l'objet dans MinIO (chemin complet)
            metadata: Métadonnées additionnelles
            
        Returns:
            URL de l'objet dans MinIO
        """
        try:
            # Ajouter des métadonnées par défaut
            if metadata is None:
                metadata = {}
            
            # Ajouter la date d'upload
            metadata['upload-date'] = datetime.now().isoformat()
            
            # Upload le fichier
            self.client.fput_object(
                self.bucket_name,
                object_name,
                local_path,
                metadata=metadata
            )
            
            # Retourner l'URL MinIO
            url = f"http://localhost:9000/{self.bucket_name}/{object_name}"
            logger.info(f"Image uploadée: {object_name}")
            return url
            
        except S3Error as e:
            logger.error(f"Erreur upload {local_path}: {e}")
            raise
    
    def upload_document_page(
        self,
        local_image_path: str,
        doc_metadata: Dict,
        page_num: int,
        total_pages: int = None
    ) -> Dict:
        """
        Upload une page de document avec métadonnées complètes.
        
        Args:
            local_image_path: Chemin local de l'image
            doc_metadata: Métadonnées du document
            page_num: Numéro de la page
            total_pages: Nombre total de pages (optionnel)
            
        Returns:
            Dict avec path et URL MinIO
        """
        # Construire le chemin dans MinIO
        object_path = self.build_object_path(doc_metadata, page_num)
        
        # Préparer les métadonnées de la page - SANS ACCENTS pour compatibilité
        page_metadata = {
            "page-number": str(page_num),
            "source-pdf": self.remove_accents(doc_metadata.get('filename', 'unknown')),
            "organisation": self.remove_accents(doc_metadata.get('organisation', 'unknown')),
            "annee": str(doc_metadata.get('annee', 'unknown')),
            "periode": self.remove_accents(doc_metadata.get('periode', 'unknown')),
            "type-document": self.remove_accents(doc_metadata.get('type_document', 'unknown'))
        }
        
        if total_pages:
            page_metadata["total-pages"] = str(total_pages)
        
        # Upload
        url = self.upload_image(local_image_path, object_path, page_metadata)
        
        return {
            "minio_path": object_path,
            "minio_url": url,
            "page_number": page_num
        }
    
    def get_image_url(self, object_name: str) -> str:
        """
        Obtient l'URL d'accès direct pour une image.
        
        Args:
            object_name: Nom de l'objet dans MinIO
            
        Returns:
            URL de l'image
        """
        return f"http://localhost:9000/{self.bucket_name}/{object_name}"
    
    def list_document_images(self, doc_metadata: Dict) -> List[str]:
        """
        Liste toutes les images d'un document spécifique.
        
        Args:
            doc_metadata: Métadonnées du document
            
        Returns:
            Liste des URLs des images
        """
        try:
            # Construire le préfixe pour ce document
            organisation = self.remove_accents(
                doc_metadata.get('organisation', 'unknown')
            ).lower().replace(' ', '_')
            
            annee = str(doc_metadata.get('annee', 'unknown'))
            
            periode = self.remove_accents(
                doc_metadata.get('periode', 'unknown')
            ).lower().replace(' ', '_')
            
            doc_id = self.generate_document_id(doc_metadata.get('filename', 'unknown'))
            
            prefix = f"{organisation}/{annee}/{periode}/{doc_id}/"
            
            # Lister les objets
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            urls = []
            for obj in objects:
                url = self.get_image_url(obj.object_name)
                urls.append(url)
                
            logger.info(f"Trouvé {len(urls)} images pour le document {doc_id}")
            return urls
            
        except S3Error as e:
            logger.error(f"Erreur listing pour {doc_metadata}: {e}")
            return []
    
    def check_document_exists(self, doc_metadata: Dict) -> bool:
        """
        Vérifie si un document existe déjà dans MinIO.
        
        Args:
            doc_metadata: Métadonnées du document
            
        Returns:
            True si le document existe déjà
        """
        images = self.list_document_images(doc_metadata)
        return len(images) > 0
    
    def delete_document(self, doc_metadata: Dict) -> int:
        """
        Supprime toutes les images d'un document.
        
        Args:
            doc_metadata: Métadonnées du document
            
        Returns:
            Nombre d'images supprimées
        """
        try:
            # Construire le préfixe
            organisation = self.remove_accents(
                doc_metadata.get('organisation', 'unknown')
            ).lower().replace(' ', '_')
            
            annee = str(doc_metadata.get('annee', 'unknown'))
            
            periode = self.remove_accents(
                doc_metadata.get('periode', 'unknown')
            ).lower().replace(' ', '_')
            
            doc_id = self.generate_document_id(doc_metadata.get('filename', 'unknown'))
            
            prefix = f"{organisation}/{annee}/{periode}/{doc_id}/"
            
            # Lister et supprimer
            objects = list(self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=True
            ))
            
            for obj in objects:
                self.client.remove_object(self.bucket_name, obj.object_name)
            
            logger.info(f"Supprimé {len(objects)} images pour {doc_id}")
            return len(objects)
            
        except S3Error as e:
            logger.error(f"Erreur suppression: {e}")
            return 0