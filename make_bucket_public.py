"""Script pour rendre le bucket MinIO public."""

from minio import Minio
import json

# Connexion à MinIO
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "rag-documents"

# Politique pour rendre le bucket public en lecture
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }
    ]
}

# Appliquer la politique
try:
    client.set_bucket_policy(bucket_name, json.dumps(policy))
    print(f"✓ Bucket '{bucket_name}' est maintenant public en lecture")
    print("Les images sont accessibles via HTTP")
    
    # Vérifier que la politique est appliquée
    current_policy = client.get_bucket_policy(bucket_name)
    print(f"\nPolitique appliquée :")
    print(json.dumps(json.loads(current_policy), indent=2))
    
except Exception as e:
    print(f"Erreur: {e}")