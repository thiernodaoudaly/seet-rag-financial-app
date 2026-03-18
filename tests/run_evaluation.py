"""
Script pour évaluer le système RAG avec RAGAS.
"""

import json
import sys
from pathlib import Path

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.opensearch_retriever import OpenSearchRetriever
from src.retrieval.reranker import ClaudeReranker
from src.generation.response_generator import ResponseGenerator

def load_test_dataset(dataset_path: str = "tests/evaluation_dataset.json"):
    """Charge le dataset de test."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']

def evaluate_rag_system():
    """Évalue le système RAG sur le dataset de test."""
    
    print("=" * 60)
    print("ÉVALUATION DU SYSTÈME RAG")
    print("=" * 60)
    
    # 1. Charger le dataset
    print("\n[1/4] Chargement du dataset de test...")
    questions = load_test_dataset()
    print(f"✓ {len(questions)} questions chargées")
    
    # 2. Initialiser les composants RAG
    print("\n[2/4] Initialisation du système RAG...")
    retriever = OpenSearchRetriever()
    reranker = ClaudeReranker()
    generator = ResponseGenerator()
    print("✓ Système initialisé")
    
    # 3. Interroger le système pour chaque question
    print("\n[3/4] Interrogation du système RAG...")
    results = []
    
    for i, item in enumerate(questions, 1):
        question = item['question']
        ground_truth = item['ground_truth']
        
        print(f"\nQuestion {i}/{len(questions)}: {question[:60]}...")
        
        try:
            # Retrieval : récupérer 10 chunks
            chunks = retriever.hybrid_search(
                question, 
                k=10,
                auto_detect_filters=True
            )
            
            # Reranking : sélectionner top 5
            chunks = reranker.rerank(question, chunks, top_k=5)
            
            # Génération : créer la réponse
            result = generator.generate_response(
                question, 
                chunks,
                include_sources=True,
                include_images=False
            )
            
            # Extraire le texte des chunks pour RAGAS
            contexts = [chunk.get('chunk_text', '') for chunk in chunks]
            
            # Préparer le résultat pour RAGAS
            results.append({
                "question": question,
                "answer": result['response'],
                "contexts": contexts,
                "ground_truth": ground_truth
            })
            
            print(f"✓ Réponse générée ({len(contexts)} contextes utilisés)")
            
        except Exception as e:
            print(f"✗ Erreur: {e}")
            results.append({
                "question": question,
                "answer": f"ERREUR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth
            })
    
    # 4. Sauvegarder les résultats
    print("\n[4/4] Sauvegarde des résultats...")
    output_path = Path("tests/evaluation_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Résultats sauvegardés dans: {output_path}")
    
    print("\n" + "=" * 60)
    print("ÉVALUATION TERMINÉE")
    print(f"{len(results)} questions traitées")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    evaluate_rag_system()