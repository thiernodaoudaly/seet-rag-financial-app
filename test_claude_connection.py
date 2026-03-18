import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Charger les variables d'environnement
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

# Vérifier que la clé existe
if not api_key:
    print("Aucune clé API trouvée dans le fichier .env")
    exit()

print(f"Clé API détectée (début): {api_key[:10]}...")

# Tester la connexion avec différents modèles
models_to_try = [
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-opus-latest",
    "claude-3-opus-20240229"
]

client = Anthropic(api_key=api_key)

for model_name in models_to_try:
    try:
        print(f"\n Test avec le modèle: {model_name}")
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Réponds juste 'OK' si tu me reçois"}
            ]
        )
        
        print(f"vSUCCÈS avec le modèle: {model_name}")
        print(f"Réponse: {response.content[0].text}")
        print(f"\n Utilisez ce modèle pour votre projet: {model_name}")
        break
        
    except Exception as e:
        print(f"Échec: {str(e)[:100]}...")
        continue
else:
    print("\n Aucun modèle n'a fonctionné. Vérifiez votre compte sur console.anthropic.com")