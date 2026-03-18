try:
    import dotenv
    import PIL
    import fitz  # PyMuPDF
    import anthropic
    import langchain
    import chainlit
    import opensearchpy
    import numpy
    import pandas
    import yaml
    print("✓ Tous les modules de base sont installés correctement")
except ImportError as e:
    print(f"Erreur d'importation : {e}")