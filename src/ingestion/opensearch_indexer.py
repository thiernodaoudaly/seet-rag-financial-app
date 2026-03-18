"""
Module pour indexer les documents dans OpenSearch v2.
Métadonnées au niveau racine
"""

from opensearchpy import OpenSearch
from typing import List, Dict
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchIndexerV2:
    """Gère l'indexation avec métadonnées au niveau racine."""
    
    def __init__(self, host: str = "localhost", port: int = 9200):
        """Initialise la connexion à OpenSearch."""
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        logger.info(f"Connexion OpenSearch V2 établie: {host}:{port}")
    
    def create_index_v2(self, index_name: str = "rag-documents-v2"):
        """
        Crée l'index avec métadonnées au niveau racine.
        """
        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 0,
                "index.knn": True,
                "index.knn.space_type": "cosinesimil",
                "analysis": {
                    "analyzer": {
                        "french_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "french_stop", "french_stemmer"]
                        }
                    },
                    "filter": {
                        "french_stop": {
                            "type": "stop",
                            "stopwords": "_french_"
                        },
                        "french_stemmer": {
                            "type": "stemmer",
                            "language": "french"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # Contenu principal
                    "content": {
                        "type": "text",
                        "analyzer": "french_analyzer"
                    },
                    
                    # Vecteur d'embedding
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    
                    # MÉTADONNÉES AU NIVEAU RACINE (changement principal)
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "organisation": {"type": "keyword"},
                    "annee": {"type": "integer"},
                    "periode": {"type": "keyword"},
                    "type_document": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_number": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "content_type": {"type": "keyword"},
                    "section": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "minio_url": {"type": "keyword"},
                    "minio_path": {"type": "keyword"},
                    "date_publication": {"type": "date"},
                    "date_indexation": {"type": "date"},
                    "keywords": {"type": "keyword"},
                    "has_numbers": {"type": "boolean"},
                    "is_complete": {"type": "boolean"}
                }
            }
        }
        
        try:
            # Supprimer l'index s'il existe
            if self.client.indices.exists(index=index_name):
                logger.warning(f"Index '{index_name}' existe déjà, suppression...")
                self.client.indices.delete(index=index_name)
            
            # Créer le nouvel index
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index V2 '{index_name}' créé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur création index V2: {e}")
            raise
    
    def migrate_from_old_index(
        self, 
        old_index: str = "rag-documents",
        new_index: str = "rag-documents-v2"
    ):
        """
        Migre les documents de l'ancien format vers le nouveau.
        """
        logger.info(f"Migration de {old_index} vers {new_index}...")
        
        # Récupérer tous les documents de l'ancien index
        query = {
            "size": 1000,  # Assez pour vos 53 documents
            "query": {"match_all": {}}
        }
        
        response = self.client.search(index=old_index, body=query)
        old_docs = response['hits']['hits']
        
        logger.info(f"Documents à migrer: {len(old_docs)}")
        
        migrated_count = 0
        for doc in old_docs:
            try:
                old_source = doc['_source']
                
                # Extraire les métadonnées de l'objet metadata
                metadata = old_source.get('metadata', {})
                
                # Créer le nouveau document avec métadonnées au niveau racine
                new_doc = {
                    "content": old_source.get('content', ''),
                    "embedding": old_source.get('embedding', []),
                    
                    # Métadonnées au niveau racine
                    "document_id": metadata.get('document_id', ''),
                    "filename": metadata.get('filename', ''),
                    "organisation": metadata.get('organisation', 'Sonatel'),
                    "annee": metadata.get('annee'),
                    "periode": metadata.get('periode', ''),
                    "type_document": metadata.get('type_document', ''),
                    "page_number": metadata.get('page_number', 0),
                    "chunk_number": metadata.get('chunk_number', 0),
                    "chunk_id": metadata.get('chunk_id', doc['_id']),
                    "content_type": metadata.get('content_type', ''),
                    "section": metadata.get('section', ''),
                    "minio_url": metadata.get('minio_url', ''),
                    "minio_path": metadata.get('minio_path', ''),
                    "date_publication": metadata.get('date_publication'),
                    "date_indexation": datetime.now().isoformat(),
                    "keywords": metadata.get('keywords', []),
                    "has_numbers": metadata.get('has_numbers', False),
                    "is_complete": metadata.get('is_complete', True)
                }
                
                # Indexer dans le nouveau format
                self.client.index(
                    index=new_index,
                    id=doc['_id'],
                    body=new_doc
                )
                
                migrated_count += 1
                
                if migrated_count % 10 == 0:
                    logger.info(f"Progression: {migrated_count}/{len(old_docs)}")
                    
            except Exception as e:
                logger.error(f"Erreur migration document {doc['_id']}: {e}")
        
        logger.info(f"Migration terminée: {migrated_count}/{len(old_docs)} documents migrés")
        return migrated_count
    
    def index_documents_v2(
        self, 
        documents: List[Dict], 
        index_name: str = "rag-documents-v2"
    ):
        """
        Indexe de nouveaux documents avec le format V2.
        """
        indexed_count = 0
        errors = []
        
        for idx, doc in enumerate(documents):
            try:
                # Les métadonnées sont dans doc['metadata']
                metadata = doc.get('metadata', {})
                
                # Créer le document avec métadonnées au niveau racine
                index_doc = {
                    "content": doc.get("content", ""),
                    "embedding": doc.get("embedding", []),
                    
                    # Métadonnées directement au niveau racine
                    "document_id": metadata.get('document_id', ''),
                    "filename": metadata.get('filename', ''),
                    "organisation": metadata.get('organisation', 'Sonatel'),
                    "annee": metadata.get('annee'),
                    "periode": metadata.get('periode', ''),
                    "type_document": metadata.get('type_document', ''),
                    "page_number": metadata.get('page_number', 0),
                    "chunk_number": metadata.get('chunk_number', 0),
                    "chunk_id": metadata.get('chunk_id', ''),
                    "content_type": metadata.get('content_type', ''),
                    "section": metadata.get('section', ''),
                    "minio_url": metadata.get('minio_url', ''),
                    "minio_path": metadata.get('minio_path', ''),
                    "date_publication": metadata.get('date_publication'),
                    "date_indexation": datetime.now().isoformat(),
                    "keywords": metadata.get('keywords', []),
                    "has_numbers": metadata.get('has_numbers', False),
                    "is_complete": metadata.get('is_complete', True)
                }
                
                # Créer l'ID unique
                doc_id = f"{metadata.get('document_id', 'unknown')}_page{metadata.get('page_number', 0)}_chunk{metadata.get('chunk_number', 0)}"
                
                # Indexer
                response = self.client.index(
                    index=index_name,
                    id=doc_id,
                    body=index_doc
                )
                
                if response.get("result") in ["created", "updated"]:
                    indexed_count += 1
                    
            except Exception as e:
                error_msg = f"Erreur indexation document {idx}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(f"Indexation V2 terminée: {indexed_count}/{len(documents)} documents")
        
        return {
            "indexed": indexed_count,
            "total": len(documents),
            "errors": errors
        }
    
    def verify_structure(self, index_name: str = "rag-documents-v2"):
        """
        Vérifie la structure d'un document pour confirmer le format.
        """
        response = self.client.search(
            index=index_name,
            body={"size": 1, "query": {"match_all": {}}}
        )
        
        if response['hits']['hits']:
            doc = response['hits']['hits'][0]['_source']
            print(f"\nStructure du document dans {index_name}:")
            print(f"Champs au niveau racine: {list(doc.keys())}")
            
            # Vérifier si periode est au niveau racine
            if 'periode' in doc:
                print(f"✓ 'periode' est au niveau racine: '{doc['periode']}'")
            else:
                print("✗ 'periode' n'est pas au niveau racine")
            
            if 'metadata' in doc:
                print("⚠ Un objet 'metadata' existe encore (ancien format)")