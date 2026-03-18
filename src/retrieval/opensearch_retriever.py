"""
Module de retrieval pour recherche hybride dans OpenSearch.
Version 2 : Compatible avec l'index V2 et détection intelligente avec Claude.
"""

import re
from typing import List, Dict, Optional
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchRetriever:
    """Effectue des recherches hybrides dans OpenSearch."""
    
    def __init__(self, host: str = "localhost", port: int = 9200):
        """Initialise le retriever."""
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        
        # Index V2
        self.index_name = "rag-documents-v2"
        
        # Claude pour comprendre les requêtes
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Modèle d'embeddings
        logger.info("Chargement du modèle BGE-M3...")
        self.encoder = SentenceTransformer("BAAI/bge-m3")
        
        if self.client.ping():
            logger.info(f"Connexion OpenSearch établie - Index: {self.index_name}")
            self._refresh_available_metadata()
        else:
            raise ConnectionError("Impossible de se connecter à OpenSearch")
    
    def _refresh_available_metadata(self):
        """Récupère TOUTES les valeurs uniques des champs filtrables."""
        try:
            # Récupérer d'abord un document pour connaître tous les champs
            sample = self.client.search(
                index=self.index_name, 
                body={"size": 1, "query": {"match_all": {}}}
            )
            
            if not sample['hits']['hits']:
                logger.warning("Aucun document dans l'index")
                self.available_metadata = {}
                return
            
            # Identifier les champs filtrables (keyword/integer)
            doc = sample['hits']['hits'][0]['_source']
            filterable_fields = ['annee', 'periode', 'organisation', 'type_document', 
                               'content_type', 'section', 'filename']
            
            # Créer les agrégations pour tous les champs filtrables
            aggs = {}
            for field in filterable_fields:
                if field in doc:
                    # Pour section qui est text+keyword, utiliser le sous-champ .keyword
                    if field == "section":
                        aggs[field] = {"terms": {"field": "section.keyword", "size": 100}}
                    else:
                        aggs[field] = {"terms": {"field": field, "size": 100}}
            
            # Exécuter les agrégations
            agg_query = {
                "size": 0,
                "aggs": aggs
            }
            
            response = self.client.search(index=self.index_name, body=agg_query)
            
            # Stocker toutes les valeurs disponibles
            self.available_metadata = {}
            for field, agg in response['aggregations'].items():
                self.available_metadata[field] = [b['key'] for b in agg['buckets']]
            
            logger.info(f"Champs filtrables disponibles: {list(self.available_metadata.keys())}")
            
        except Exception as e:
            logger.warning(f"Impossible de récupérer les métadonnées: {e}")
            self.available_metadata = {}
    
    def encode_query(self, query: str) -> List[float]:
        """Encode une requête en vecteur."""
        embedding = self.encoder.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def extract_filters_from_query(self, query: str) -> Dict:
        """
        Utilise Claude pour extraire TOUS les filtres possibles de la requête.
        """
        if not hasattr(self, 'available_metadata') or not self.available_metadata:
            self._refresh_available_metadata()
        
        # Construire le prompt avec TOUTES les métadonnées disponibles
        metadata_description = ""
        for field, values in self.available_metadata.items():
            if values:
                # Limiter à 10 valeurs pour ne pas surcharger le prompt
                sample_values = values[:10]
                if len(values) > 10:
                    sample_values.append("...")
                metadata_description += f"- {field}: {sample_values}\n"
        
        prompt = f"""Analyse cette requête et extrais TOUS les filtres pertinents.

Requête : "{query}"

Champs disponibles avec leurs valeurs :
{metadata_description}

Instructions :
- Détecte TOUS les champs mentionnés dans la requête
- Pour les périodes : "premier trimestre" = "T1", "semestre 1" = "S1", etc.
- Pour les types : "rapport d'activités" peut être mentionné directement
- N'EXTRAIS JAMAIS le champ "organisation" (inutile, tous les documents sont du même groupe)
- Retourne un JSON avec TOUS les filtres détectés SAUF organisation
- Si aucun filtre, retourne {{}}

Exemples de retour :
{{"annee": 2024, "periode": "T1 2024", "type_document": "rapport d'activités"}}
{{"annee": 2023, "type_document": "résultats consolidés"}}

Retourne UNIQUEMENT le JSON, sans explication."""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            filters = json.loads(response.content[0].text.strip())
            
            # Valider que les valeurs existent vraiment dans l'index
            validated_filters = {}
            for field, value in filters.items():
                if field in self.available_metadata:
                    # Pour les strings, vérifier la correspondance exacte
                    if isinstance(value, str):
                        if value in self.available_metadata[field]:
                            validated_filters[field] = value
                        else:
                            # Essayer une correspondance insensible à la casse
                            for available_value in self.available_metadata[field]:
                                if available_value.lower() == value.lower():
                                    validated_filters[field] = available_value
                                    break
                    else:
                        # Pour les nombres (comme annee)
                        if value in self.available_metadata[field]:
                            validated_filters[field] = value
            
            if validated_filters:
                logger.info(f"Filtres validés: {validated_filters}")
            
            return validated_filters
            
        except Exception as e:
            logger.error(f"Erreur extraction filtres: {e}")
            return {}
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        auto_detect_filters: bool = True
    ) -> List[Dict]:
        """
        Recherche hybride : vectorielle + textuelle.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats
            filters: Filtres manuels (optionnel)
            auto_detect_filters: Détection automatique des filtres
            
        Returns:
            Liste des chunks pertinents
        """
        # Détection automatique si activée
        if auto_detect_filters and not filters:
            filters = self.extract_filters_from_query(query)
        
        # Encoder la requête
        query_vector = self.encode_query(query)
        
        # Construire les filtres
        must_clauses = []
        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})
        
        # Requête hybride
        hybrid_query = {
            "size": k * 2,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": k
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", "section^2"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        
        # Exécuter la recherche
        response = self.client.search(index=self.index_name, body=hybrid_query)
        
        # Traiter les résultats
        results = []
        seen_ids = set()
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            doc_id = source.get('chunk_id', hit['_id'])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            results.append({
                "chunk_id": doc_id,
                "chunk_text": source.get('content', ''),
                "page_number": source.get('page_number', 0),
                "content_type": source.get('content_type', ''),
                "section": source.get('section', ''),
                "minio_url": source.get('minio_url', ''),
                "filename": source.get('filename', ''),
                "periode": source.get('periode', ''),
                "annee": source.get('annee', ''),
                "organisation": source.get('organisation', ''),
                "type_document": source.get('type_document', ''),
                "score": hit['_score'],
                "metadata": source
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Recherche: {len(results)} chunks trouvés")
        return results[:k]
    
    def search_by_period(
        self,
        query: str,
        annee: Optional[int] = None,
        periode: Optional[str] = None,
        organisation: Optional[str] = None,
        k: int = 5
    ) -> List[Dict]:
        """Recherche avec filtres explicites."""
        filters = {}
        
        if annee:
            filters["annee"] = annee
        if periode:
            filters["periode"] = periode
        if organisation:
            filters["organisation"] = organisation
        
        return self.hybrid_search(query, k=k, filters=filters, auto_detect_filters=False)