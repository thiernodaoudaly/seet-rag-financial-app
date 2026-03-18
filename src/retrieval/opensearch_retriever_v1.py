"""
Module de retrieval pour recherche hybride dans OpenSearch.
Combine recherche vectorielle (HNSW) et textuelle (BM25).
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchRetriever:
    """Effectue des recherches hybrides dans OpenSearch."""
    
    def __init__(self, host: str = "localhost", port: int = 9200):
        """
        Initialise le retriever.
        
        Args:
            host: Hôte OpenSearch
            port: Port OpenSearch
        """
        # Connexion OpenSearch
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        
        # INDEX UNIQUE pour tous les documents
        self.index_name = "rag-documents"
        
        # Modèle d'embeddings (même que pour l'indexation)
        logger.info("Chargement du modèle BGE-M3...")
        self.encoder = SentenceTransformer("BAAI/bge-m3")
        
        if self.client.ping():
            logger.info(f"Connexion OpenSearch établie - Index: {self.index_name}")
            # Récupérer les métadonnées disponibles dans l'index
            self._refresh_available_metadata()
        else:
            raise ConnectionError("Impossible de se connecter à OpenSearch")
    
    def _refresh_available_metadata(self):
        """
        Récupère les valeurs uniques des métadonnées depuis l'index.
        Cela permet de savoir quelles périodes, années, organisations sont disponibles.
        """
        try:
            # Agrégations pour obtenir les valeurs uniques
            agg_query = {
                "size": 0,
                "aggs": {
                    "annees": {
                        "terms": {"field": "metadata.annee", "size": 100}
                    },
                    "periodes": {
                        "terms": {"field": "metadata.periode", "size": 100}
                    },
                    "organisations": {
                        "terms": {"field": "metadata.organisation", "size": 100}
                    },
                    "types_document": {
                        "terms": {"field": "metadata.type_document", "size": 100}
                    }
                }
            }
            
            response = self.client.search(index=self.index_name, body=agg_query)
            
            self.available_metadata = {
                'annees': [b['key'] for b in response['aggregations']['annees']['buckets']],
                'periodes': [b['key'] for b in response['aggregations']['periodes']['buckets']],
                'organisations': [b['key'] for b in response['aggregations']['organisations']['buckets']],
                'types_document': [b['key'] for b in response['aggregations']['types_document']['buckets']]
            }
            
            logger.info(f"Métadonnées disponibles: {self.available_metadata}")
            
        except Exception as e:
            logger.warning(f"Impossible de récupérer les métadonnées: {e}")
            self.available_metadata = {}
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode une requête en vecteur.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Vecteur d'embedding
        """
        embedding = self.encoder.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def extract_filters_from_query(self, query: str) -> Dict:
        """
        Extrait intelligemment les filtres depuis la requête en utilisant
        les métadonnées réellement disponibles dans l'index.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Dictionnaire de filtres à appliquer
        """
        filters = {}
        query_lower = query.lower()
        
        # Rafraîchir les métadonnées disponibles
        if not hasattr(self, 'available_metadata') or not self.available_metadata:
            self._refresh_available_metadata()
        
        # 1. Détecter l'année parmi celles disponibles
        for annee in self.available_metadata.get('annees', []):
            if str(annee) in query:
                filters['annee'] = annee
                logger.info(f"Année détectée dans la requête: {annee}")
                break
        
        # 2. Détecter la période parmi celles disponibles
        for periode in self.available_metadata.get('periodes', []):
            # Normaliser pour la comparaison (T1 2024 -> t1, 2024)
            periode_normalized = periode.lower().replace(' ', '')
            query_normalized = query_lower.replace(' ', '')
            
            # Vérifier si la période est mentionnée
            if any(term in query_normalized for term in [
                periode_normalized,
                periode.lower(),
                periode.replace(' ', '_').lower()
            ]):
                filters['periode'] = periode
                logger.info(f"Période détectée dans la requête: {periode}")
                break
        
        # 3. Détecter l'organisation si mentionnée
        for org in self.available_metadata.get('organisations', []):
            if org.lower() in query_lower:
                filters['organisation'] = org
                logger.info(f"Organisation détectée: {org}")
                break
        
        # 4. Détecter des mots-clés pour cibler certains types de contenu
        content_keywords = {
            'chiffre': ['kpi_financiers', 'tableau_financier', 'compte_resultat'],
            'revenue': ['kpi_financiers', 'tableau_financier'],
            'capex': ['tableau_financier', 'kpi_financiers'],
            'investissement': ['tableau_financier', 'kpi_financiers'],
            'résultat': ['compte_resultat', 'kpi_financiers'],
            'ebitda': ['compte_resultat', 'kpi_financiers'],
            'bilan': ['bilan'],
            'situation': ['faits_marquants', 'kpi_financiers'],
            'pays': ['faits_marquants', 'tableau_financier']
        }
        
        # Chercher les mots-clés dans la requête
        preferred_types = []
        for keyword, types in content_keywords.items():
            if keyword in query_lower:
                preferred_types.extend(types)
        
        if preferred_types:
            # On ne filtre pas strictement mais on va booster ces types
            filters['_preferred_content_types'] = list(set(preferred_types))
            logger.info(f"Types de contenu préférés: {filters['_preferred_content_types']}")
        
        return filters
    
    def vector_search(
        self, 
        query: str, 
        index_name: str = None,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Recherche par similarité vectorielle avec HNSW.
        
        Args:
            query: Question de l'utilisateur
            index_name: Index à rechercher
            k: Nombre de résultats
            filters: Filtres sur métadonnées
            
        Returns:
            Liste des chunks pertinents
        """
        if index_name is None:
            index_name = self.index_name
            
        # Encoder la requête
        query_vector = self.encode_query(query)
        
        # Construire la requête k-NN
        knn_query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        # Ajouter des filtres si spécifiés
        if filters:
            filter_conditions = []
            for field, value in filters.items():
                if not field.startswith('_'):  # Ignorer les filtres internes
                    filter_conditions.append({
                        "term": {f"metadata.{field}": value}
                    })
            
            if filter_conditions:
                knn_query["query"]["knn"]["embedding"]["filter"] = {
                    "bool": {
                        "must": filter_conditions
                    }
                }
        
        # Exécuter la recherche
        response = self.client.search(index=index_name, body=knn_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            metadata = source.get('metadata', {})
            
            results.append({
                "chunk_id": metadata.get('chunk_id', hit['_id']),
                "chunk_text": source.get('content', ''),
                "page_number": metadata.get('page_number', 0),
                "content_type": metadata.get('content_type', ''),
                "section": metadata.get('section', ''),
                "minio_url": metadata.get('minio_url', ''),
                "filename": metadata.get('filename', ''),
                "periode": metadata.get('periode', ''),
                "annee": metadata.get('annee', ''),
                "organisation": metadata.get('organisation', ''),
                "score": hit['_score'],
                "metadata": metadata
            })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        index_name: str = None,
        k: int = 5,
        filters: Optional[Dict] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        auto_detect_filters: bool = True
    ) -> List[Dict]:
        """
        Recherche hybride : vectorielle + textuelle avec détection automatique du contexte.
        
        Args:
            query: Question de l'utilisateur
            index_name: Index à rechercher
            k: Nombre de résultats
            filters: Filtres manuels sur métadonnées
            vector_weight: Poids de la recherche vectorielle
            text_weight: Poids de la recherche textuelle
            auto_detect_filters: Détection automatique des filtres depuis la requête
            
        Returns:
            Liste des chunks pertinents avec score combiné
        """
        if index_name is None:
            index_name = self.index_name
        
        # Détection automatique des filtres si activée
        if auto_detect_filters and not filters:
            filters = self.extract_filters_from_query(query)
            if filters:
                logger.info(f"Filtres auto-détectés: {filters}")
        
        # Extraire les types préférés
        preferred_types = []
        if filters and '_preferred_content_types' in filters:
            preferred_types = filters.pop('_preferred_content_types')
        
        # Encoder la requête
        query_vector = self.encode_query(query)
        
        # Construire les clauses de filtre strictes
        must_clauses = []
        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {f"metadata.{field}": value}})
        
        # Augmenter k si on a des filtres (pour avoir assez de résultats)
        search_k = k * 2 if filters else k
        
        # Requête hybride
        hybrid_query = {
            "size": search_k * 2,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        # Recherche vectorielle
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": search_k
                                }
                            }
                        },
                        # Recherche textuelle
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", "metadata.section^2", "metadata.keywords"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        # Exécuter la recherche
        response = self.client.search(index=index_name, body=hybrid_query)
        
        # Traiter les résultats avec scoring amélioré
        results = []
        seen_ids = set()
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            metadata = source.get('metadata', {})
            
            # ID unique pour éviter les doublons
            doc_id = metadata.get('chunk_id', hit['_id'])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            # Score de base
            score = hit['_score']
            
            # Booster si le type de contenu est préféré
            if preferred_types and metadata.get('content_type') in preferred_types:
                score *= 1.5
                logger.debug(f"Boost pour type préféré: {metadata.get('content_type')}")
            
            results.append({
                "chunk_id": doc_id,
                "chunk_text": source.get('content', ''),
                "page_number": metadata.get('page_number', 0),
                "content_type": metadata.get('content_type', ''),
                "section": metadata.get('section', ''),
                "minio_url": metadata.get('minio_url', ''),
                "filename": metadata.get('filename', ''),
                "periode": metadata.get('periode', ''),
                "annee": metadata.get('annee', ''),
                "organisation": metadata.get('organisation', ''),
                "score": score,
                "metadata": metadata
            })
        
        # Trier par score et limiter
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def search_by_content_type(
        self,
        query: str,
        content_types: List[str],
        k: int = 5
    ) -> List[Dict]:
        """
        Recherche limitée à certains types de contenu.
        
        Args:
            query: Question de l'utilisateur
            content_types: Types de contenu à rechercher
            k: Nombre de résultats
            
        Returns:
            Liste des chunks pertinents
        """
        # Utiliser la recherche hybride avec boost sur les types
        filters = {'_preferred_content_types': content_types}
        return self.hybrid_search(query, k=k, filters=filters)
    
    def search_by_period(
        self,
        query: str,
        annee: Optional[int] = None,
        periode: Optional[str] = None,
        organisation: Optional[str] = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Recherche avec filtres temporels explicites.
        
        Args:
            query: Question de l'utilisateur
            annee: Année à filtrer
            periode: Période à filtrer
            organisation: Organisation à filtrer
            k: Nombre de résultats
            
        Returns:
            Liste des chunks pertinents
        """
        filters = {}
        
        if annee:
            filters["annee"] = annee
        if periode:
            filters["periode"] = periode
        if organisation:
            filters["organisation"] = organisation
        
        return self.hybrid_search(query, k=k, filters=filters, auto_detect_filters=False)
    
    def search_multi_period(
        self,
        query: str,
        periods: List[Dict],
        k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Recherche sur plusieurs périodes pour comparaisons.
        
        Args:
            query: Question de l'utilisateur
            periods: Liste de dictionnaires avec annee/periode
            k: Nombre de résultats par période
            
        Returns:
            Dictionnaire {période: résultats}
        """
        results_by_period = {}
        
        for period_filter in periods:
            period_key = f"{period_filter.get('annee', '')}_{period_filter.get('periode', '')}"
            results = self.search_by_period(
                query,
                annee=period_filter.get('annee'),
                periode=period_filter.get('periode'),
                k=k
            )
            results_by_period[period_key] = results
        
        return results_by_period