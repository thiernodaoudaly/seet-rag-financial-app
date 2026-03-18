"""
Script pour évaluer avec RAGAS les résultats du système RAG.
Utilise Claude 4 Sonnet pour l'évaluation et BGE-M3 pour les embeddings.
"""

import json
import os
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_evaluation_results(results_path: str = "tests/evaluation_results.json"):
    """Charge les résultats de l'évaluation."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_ragas_dataset(results):
    """Prépare les données au format RAGAS."""
    valid_results = [r for r in results if r['contexts']]
    data = {
        "question": [r["question"] for r in valid_results],
        "answer": [r["answer"] for r in valid_results],
        "contexts": [r["contexts"] for r in valid_results],
        "ground_truth": [r["ground_truth"] for r in valid_results]
    }
    return Dataset.from_dict(data)

def run_ragas_evaluation():
    """Lance l'évaluation RAGAS avec Claude 4 Sonnet."""
    
    print("=" * 60)
    print("EVALUATION RAGAS DU SYSTEME RAG")
    print("Evaluateur: Claude 4 Sonnet")
    print("=" * 60)
    
    print("\n[1/5] Chargement des résultats...")
    results = load_evaluation_results()
    print(f"OK - {len(results)} résultats chargés")
    
    print("\n[2/5] Préparation du dataset RAGAS...")
    dataset = prepare_ragas_dataset(results)
    print(f"OK - {len(dataset)} exemples valides (avec contextes)")
    
    print("\n[3/5] Configuration de Claude 4 Sonnet pour RAGAS...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERREUR: ANTHROPIC_API_KEY non trouvée dans .env")
        return None
    
    evaluator_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key,
        temperature=0,
        max_tokens=4096
    )
    print("OK - Claude 4 Sonnet configuré comme évaluateur")
    
    print("\n[4/5] Configuration des embeddings BGE-M3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("OK - Embeddings BGE-M3 configurés (même que le retriever)")
    
    print("\n[5/5] Évaluation avec RAGAS...")
    print("ATTENTION: Cette étape prend 5-10 minutes")
    print("           (RAGAS utilise Claude 4 pour évaluer chaque réponse)")
    print("")
    
    try:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=embeddings
        )
        
        # Fonction helper pour extraire les scores
        def get_score(value):
            if isinstance(value, list):
                valid_scores = [s for s in value if s is not None and not (isinstance(s, float) and s != s)]
                return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            return float(value) if value is not None else 0.0
        
        # Extraire les scores (utiliser [] au lieu de .get())
        cp_score = get_score(result['context_precision'])
        cr_score = get_score(result['context_recall'])
        f_score = get_score(result['faithfulness'])
        ar_score = get_score(result['answer_relevancy'])
        
        # Afficher les résultats
        print("\n" + "=" * 60)
        print("RESULTATS DE L'EVALUATION RAGAS")
        print("=" * 60)
        
        print("\n[RETRIEVAL - Qualité des chunks récupérés]")
        print(f"\nContext Precision:")
        print(f"  Score: {cp_score:.3f} / 1.000")
        
        print(f"\nContext Recall:")
        print(f"  Score: {cr_score:.3f} / 1.000")
        
        retrieval_score = (cp_score + cr_score) / 2
        print(f"\n  >> Score moyen RETRIEVAL: {retrieval_score:.3f}")
        
        print("\n" + "-" * 60)
        print("\n[GENERATION - Qualité des réponses générées]")
        
        print(f"\nFaithfulness:")
        print(f"  Score: {f_score:.3f} / 1.000")
        
        print(f"\nAnswer Relevancy:")
        print(f"  Score: {ar_score:.3f} / 1.000")
        
        generation_score = (f_score + ar_score) / 2
        print(f"\n  >> Score moyen GENERATION: {generation_score:.3f}")
        
        print("\n" + "=" * 60)
        
        avg_score = (f_score + ar_score + cp_score + cr_score) / 4
        print(f"\nSCORE GLOBAL DU SYSTEME RAG: {avg_score:.3f} / 1.000")
        print("=" * 60)
        
        print("\nINTERPRETATION:")
        if avg_score >= 0.8:
            print("  >> Excellent système RAG")
        elif avg_score >= 0.6:
            print("  >> Bon système RAG avec marge d'amélioration")
        elif avg_score >= 0.4:
            print("  >> Système correct mais nécessite des améliorations")
        else:
            print("  >> Système nécessite des améliorations importantes")
        
        print("\nPOINTS D'ATTENTION:")
        if cp_score < 0.7:
            print("  - Retrieval: Améliorer la précision")
        if cr_score < 0.7:
            print("  - Retrieval: Améliorer le rappel")
        if f_score < 0.8:
            print("  - Generation: Réduire les hallucinations")
        if ar_score < 0.7:
            print("  - Generation: Améliorer la pertinence")
        
        if all(s >= 0.7 for s in [f_score, ar_score, cp_score, cr_score]):
            print("  - Aucun point d'attention majeur. Système performant.")
        
        output_path = Path("tests/ragas_scores.json")
        result_dict = {
            'context_precision': cp_score,
            'context_recall': cr_score,
            'faithfulness': f_score,
            'answer_relevancy': ar_score,
            'retrieval_score': retrieval_score,
            'generation_score': generation_score,
            'global_score': avg_score
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nScores sauvegardés dans: {output_path}")
        return result
        
    except Exception as e:
        print(f"\nERREUR lors de l'évaluation RAGAS: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_ragas_evaluation()