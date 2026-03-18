"""
Script de migration vers la nouvelle structure OpenSearch.
Migre de rag-documents vers rag-documents-v2 avec métadonnées au niveau racine.
"""

from src.ingestion.opensearch_indexer import OpenSearchIndexerV2
from opensearchpy import OpenSearch

def main():
    print("="*60)
    print("MIGRATION OPENSEARCH V1 → V2")
    print("="*60)
    
    # Initialiser l'indexeur V2
    indexer = OpenSearchIndexerV2()
    
    # 1. Créer le nouvel index
    print("\n[1/4] Création du nouvel index 'rag-documents-v2'...")
    indexer.create_index_v2("rag-documents-v2")
    
    # 2. Migration des documents
    print("\n[2/4] Migration des documents...")
    migrated = indexer.migrate_from_old_index(
        old_index="rag-documents",
        new_index="rag-documents-v2"
    )
    
    # 3. Vérification de la structure
    print("\n[3/4] Vérification de la nouvelle structure...")
    indexer.verify_structure("rag-documents-v2")
    
    # 4. Test de requête directe
    print("\n[4/4] Test de requête sur 'periode' au niveau racine...")
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False
    )
    
    test_response = client.search(
        index="rag-documents-v2",
        body={
            "query": {"term": {"periode": "T1 2024"}},
            "size": 2
        }
    )
    
    print(f"Documents trouvés avec periode='T1 2024': {test_response['hits']['total']['value']}")
    
    print("\n" + "="*60)
    print("MIGRATION TERMINÉE")
    print(f"Documents migrés: {migrated}")
    print("Nouvel index: rag-documents-v2")
    print("="*60)

if __name__ == "__main__":
    main()