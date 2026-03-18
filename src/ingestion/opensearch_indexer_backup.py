"""
Module pour indexer les documents dans OpenSearch.
Un seul index 'rag-documents' pour tous les documents, avec métadonnées pour filtrer.
"""

from opensearchpy import OpenSearch
from typing import List, Dict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchIndexer:
    """Gère l'indexation des documents dans OpenSearch."""
    
    def __init__(self, host: str = "localhost", port: int = 9200):
        """Initialise la connexion à OpenSearch."""
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        logger.info(f"Connexion OpenSearch établie: {host}:{port}")
    
    def create_index(self, index_name: str = "rag-documents"):
        """
        Crée l'index avec le mapping approprié pour RAG multimodal.
        Un seul index pour tous les documents.
        """
        # Mapping optimisé pour recherche hybride et filtrage
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
                        "analyzer": "french_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Vecteur d'embedding (BGE-M3 = 1024 dimensions)
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
                    
                    # Métadonnées pour filtrage et organisation
                    "metadata": {
                        "properties": {
                            # Identification du document
                            "document_id": {"type": "keyword"},
                            "filename": {"type": "keyword"},
                            
                            # Organisation hiérarchique (comme MinIO)
                            "organisation": {"type": "keyword"},
                            "annee": {"type": "integer"},
                            "periode": {"type": "keyword"},
                            "type_document": {"type": "keyword"},
                            
                            # Localisation dans le document
                            "page_number": {"type": "integer"},
                            "chunk_number": {"type": "integer"},
                            "total_chunks": {"type": "integer"},
                            
                            # URL MinIO de la page source
                            "minio_url": {"type": "keyword"},
                            "minio_path": {"type": "keyword"},
                            
                            # Dates
                            "date_publication": {"type": "date"},
                            "date_indexation": {"type": "date"},
                            
                            # Statistiques
                            "chunk_size": {"type": "integer"},
                            "confidence_score": {"type": "float"}
                        }
                    }
                }
            }
        }
        
        try:
            # Supprimer l'index s'il existe déjà
            if self.client.indices.exists(index=index_name):
                logger.warning(f"Index '{index_name}' existe déjà, suppression...")
                self.client.indices.delete(index=index_name)
            
            # Créer le nouvel index
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index '{index_name}' créé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur création index: {e}")
            raise
    
    def index_documents(
        self, 
        documents: List[Dict], 
        index_name: str = "rag-documents"
    ):
        """
        Indexe une liste de documents avec leurs embeddings et métadonnées.
        
        Args:
            documents: Liste de documents avec content, embedding et metadata
            index_name: Nom de l'index (par défaut: rag-documents)
        """
        indexed_count = 0
        errors = []
        
        for idx, doc in enumerate(documents):
            try:
                # Préparer le document pour l'indexation
                index_doc = {
                    "content": doc.get("content", ""),
                    "embedding": doc.get("embedding", []),
                    "metadata": doc.get("metadata", {})
                }
                
                # Ajouter date d'indexation
                from datetime import datetime
                index_doc["metadata"]["date_indexation"] = datetime.now().isoformat()
                
                # Créer un ID unique basé sur les métadonnées
                doc_id = f"{doc['metadata'].get('document_id', 'unknown')}_page{doc['metadata'].get('page_number', 0)}_chunk{doc['metadata'].get('chunk_number', 0)}"
                
                # Indexer le document
                response = self.client.index(
                    index=index_name,
                    id=doc_id,
                    body=index_doc
                )
                
                if response.get("result") in ["created", "updated"]:
                    indexed_count += 1
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Progression: {idx + 1}/{len(documents)} documents indexés")
                
            except Exception as e:
                error_msg = f"Erreur indexation document {idx}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(f"Indexation terminée: {indexed_count}/{len(documents)} documents indexés")
        if errors:
            logger.warning(f"{len(errors)} erreurs rencontrées")
        
        return {
            "indexed": indexed_count,
            "total": len(documents),
            "errors": errors
        }
    
    def search_hybrid(
        self, 
        query: str,
        query_embedding: List[float],
        index_name: str = "rag-documents",
        filters: Dict = None,
        size: int = 5,
        text_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        """
        Recherche hybride : textuelle (BM25) + vectorielle (KNN).
        
        Args:
            query: Requête textuelle
            query_embedding: Vecteur de la requête
            index_name: Nom de l'index
            filters: Filtres sur les métadonnées (ex: {"organisation": "sonatel", "annee": 2025})
            size: Nombre de résultats
            text_weight: Poids de la recherche textuelle
            vector_weight: Poids de la recherche vectorielle
        """
        # Construire les filtres si spécifiés
        filter_queries = []
        if filters:
            for key, value in filters.items():
                filter_queries.append({
                    "term": {f"metadata.{key}": value}
                })
        
        # Requête textuelle BM25
        text_query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "analyzer": "french_analyzer"
                            }
                        }
                    }
                ]
            }
        }
        
        # Ajouter les filtres à la requête textuelle
        if filter_queries:
            text_query["bool"]["filter"] = filter_queries
        
        # Requête vectorielle KNN
        knn_query = {
            "size": size * 2,  # Récupérer plus pour le reranking
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": size * 2
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Ajouter les filtres à la requête KNN
        if filter_queries:
            knn_query["query"]["bool"]["filter"] = filter_queries
        
        try:
            # Exécuter recherche textuelle
            text_results = self.client.search(
                index=index_name,
                body={
                    "query": text_query,
                    "size": size * 2,
                    "_source": ["content", "metadata"]
                }
            )
            
            # Exécuter recherche vectorielle
            vector_results = self.client.search(
                index=index_name,
                body=knn_query
            )
            
            # Combiner et scorer les résultats
            combined_results = self._combine_results(
                text_results, 
                vector_results,
                text_weight, 
                vector_weight
            )
            
            # Retourner les top K résultats
            return combined_results[:size]
            
        except Exception as e:
            logger.error(f"Erreur recherche hybride: {e}")
            return []
    
    def _combine_results(
        self, 
        text_results, 
        vector_results, 
        text_weight, 
        vector_weight
    ):
        """Combine les résultats de recherche textuelle et vectorielle."""
        scores = {}
        documents = {}
        
        # Traiter résultats textuels
        for hit in text_results.get("hits", {}).get("hits", []):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0) + (hit["_score"] * text_weight)
            documents[doc_id] = hit["_source"]
        
        # Traiter résultats vectoriels
        for hit in vector_results.get("hits", {}).get("hits", []):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0) + (hit["_score"] * vector_weight)
            if doc_id not in documents:
                documents[doc_id] = hit["_source"]
        
        # Trier par score combiné
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Formater les résultats
        results = []
        for doc_id, score in sorted_docs:
            result = documents[doc_id]
            result["_score"] = score
            result["_id"] = doc_id
            results.append(result)
        
        return results
    
    def get_stats(self, index_name: str = "rag-documents"):
        """Obtient les statistiques de l'index."""
        try:
            stats = self.client.indices.stats(index=index_name)
            count = self.client.count(index=index_name)
            
            return {
                "total_documents": count["count"],
                "index_size": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "index_size_mb": round(stats["indices"][index_name]["total"]["store"]["size_in_bytes"] / (1024*1024), 2)
            }
        except Exception as e:
            logger.error(f"Erreur récupération stats: {e}")
            return {}