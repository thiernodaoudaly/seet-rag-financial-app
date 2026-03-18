"""
Interface Streamlit pour l'application RAG.
Compatible avec l'index V2 et la nouvelle architecture.
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import time

# Import des modules du pipeline
from src.ingestion.pdf_to_images import PDFToImageConverter
from src.ingestion.multimodal_extractor import MultimodalExtractor
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.opensearch_indexer import OpenSearchIndexerV2
from src.retrieval.opensearch_retriever import OpenSearchRetriever
from src.retrieval.reranker import ClaudeReranker
from src.generation.response_generator import ResponseGenerator

# Configuration de la page
st.set_page_config(
    page_title="Seet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec icônes Material Icons
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    /* Boutons principaux */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sources */
    .source-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 12px;
        margin: 8px 0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .source-title {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 4px;
    }
    
    .source-details {
        color: #7f8c8d;
        font-size: 0.9em;
    }
    
    /* Images */
    .image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .image-caption {
        background: #f8f9fa;
        padding: 8px;
        font-size: 0.85em;
        color: #495057;
        border-top: 1px solid #dee2e6;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stat-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Header avec icône */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
    }
    
    .section-header .material-icons {
        color: #667eea;
        font-size: 28px;
    }
    
    /* Tips box */
    .tips-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .tips-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 8px;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============= DÉFINITION DES FONCTIONS =============

def process_pdf(uploaded_file):
    """Traite un PDF uploadé avec le pipeline complet."""
    
    progress_container = st.sidebar.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status = st.empty()
        
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp())
            pdf_path = temp_dir / uploaded_file.name
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Extraction métadonnées
            status.info("Extraction des métadonnées du document...")
            progress_bar.progress(10)
            
            extractor = MultimodalExtractor()
            doc_metadata = extractor.extract_metadata_from_filename(uploaded_file.name)
            doc_metadata["document_id"] = Path(uploaded_file.name).stem
            
            with st.sidebar.expander("Métadonnées extraites", expanded=False):
                st.json(doc_metadata)
            
            # 2. Conversion PDF → MinIO
            status.info("Conversion des pages en images...")
            progress_bar.progress(20)
            
            converter = PDFToImageConverter()
            pages_info = converter.convert_pdf_to_minio(str(pdf_path), doc_metadata)
            
            # 3. Extraction du contenu
            status.info(f"Extraction de {len(pages_info)} pages avec Claude Vision...")
            progress_bar.progress(40)
            
            output_dir = Path("data/processed/extractions_minio") / doc_metadata.get('filename', 'unknown')
            extraction_result = extractor.extract_all_pages_from_minio(pages_info, output_dir)
            
            # 4. Chunking sémantique
            status.info("Découpage sémantique du contenu...")
            progress_bar.progress(60)
            
            chunker = SemanticChunker()
            all_chunks = []
            
            for extraction in extraction_result['extractions']:
                if extraction['status'] == 'success':
                    chunk_metadata = {
                        **doc_metadata,
                        "page_number": extraction['page_number'],
                        "minio_url": extraction.get('minio_url', ''),
                        "minio_path": pages_info[extraction['page_number']-1].get('minio_path', ''),
                        "document_id": doc_metadata.get('document_id', 'unknown'),
                        "filename": doc_metadata.get('filename', 'unknown')
                    }
                    
                    page_chunks = chunker.create_chunks(
                        extraction['content'],
                        chunk_metadata
                    )
                    
                    for idx, chunk in enumerate(page_chunks):
                        chunk['metadata']['chunk_number'] = idx + 1
                        chunk['metadata']['total_chunks'] = len(page_chunks)
                    
                    all_chunks.extend(page_chunks)
            
            # 5. Embeddings
            status.info("Génération des embeddings BGE-M3...")
            progress_bar.progress(80)
            
            embedder = EmbeddingGenerator()
            chunks_with_embeddings = embedder.generate_embeddings(all_chunks)
            
            # 6. Indexation
            status.info("Indexation dans OpenSearch...")
            progress_bar.progress(90)
            
            result = st.session_state.indexer.index_documents_v2(chunks_with_embeddings)
            
            progress_bar.progress(100)
            status.success("Document indexé avec succès!")
            
            # Statistiques
            with st.sidebar.expander("Statistiques du traitement", expanded=True):
                st.markdown(f"""
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div class='stat-card'>
                        <div class='stat-value'>{len(pages_info)}</div>
                        <div class='stat-label'>Pages traitées</div>
                    </div>
                    <div class='stat-card'>
                        <div class='stat-value'>{len(all_chunks)}</div>
                        <div class='stat-label'>Chunks créés</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.refresh_stats()
            time.sleep(3)
            progress_bar.empty()
            status.empty()
            st.rerun()
            
        except Exception as e:
            status.error(f"Erreur: {str(e)}")
            st.sidebar.error(f"Détails: {str(e)}")
            
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

def display_sources(sources):
    """Affiche les sources de manière formatée."""
    if sources:
        unique_docs = {}
        for source in sources:
            filename = source.get('filename', 'Document')
            if filename not in unique_docs:
                unique_docs[filename] = []
            unique_docs[filename].append(source)
        
        for filename, doc_sources in unique_docs.items():
            st.markdown(f"**{filename}**")
            for source in doc_sources:
                periode = source.get('periode', '')
                annee = source.get('annee', '')
                type_doc = source.get('type_document', '')
                time_info = f" ({periode} {annee})" if periode or annee else ""
                
                st.markdown(f"""
                <div class='source-card'>
                    <div class='source-title'>Page {source['page']}: {source.get('section', 'N/A')}</div>
                    <div class='source-details'>Type: {source.get('type', type_doc)}{time_info}</div>
                </div>
                """, unsafe_allow_html=True)

def display_images(images):
    """Affiche TOUTES les images des pages citées avec légendes détaillées."""
    if images:
        # Afficher toutes les images en grille de 2 colonnes
        for idx in range(0, len(images), 2):
            cols = st.columns(2)
            
            # Première colonne
            with cols[0]:
                img = images[idx]
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(img["url"], use_container_width=True)
                
                caption_parts = [f"Document: {img.get('filename', 'N/A')}"]
                caption_parts.append(f"Page {img['page']}")
                
                if img.get('type'):
                    caption_parts.append(f"Type: {img['type']}")
                if img.get('periode'):
                    caption_parts.append(f"Période: {img['periode']}")
                
                st.markdown(f"""
                <div class='image-caption'>{' | '.join(caption_parts)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Deuxième colonne (si elle existe)
            if idx + 1 < len(images):
                with cols[1]:
                    img = images[idx + 1]
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    st.image(img["url"], use_container_width=True)
                    
                    caption_parts = [f"Document: {img.get('filename', 'N/A')}"]
                    caption_parts.append(f"Page {img['page']}")
                    
                    if img.get('type'):
                        caption_parts.append(f"Type: {img['type']}")
                    if img.get('periode'):
                        caption_parts.append(f"Période: {img['periode']}")
                    
                    st.markdown(f"""
                    <div class='image-caption'>{' | '.join(caption_parts)}</div>
                    </div>
                    """, unsafe_allow_html=True)

def refresh_stats():
    """Rafraîchit les statistiques depuis l'index."""
    try:
        response = st.session_state.indexer.client.count(index="rag-documents-v2")
        st.session_state.total_docs = response.get('count', 0)
    except:
        st.session_state.total_docs = 0

# ============= INITIALISATION =============

if 'initialized' not in st.session_state:
    with st.spinner("Initialisation du système RAG..."):
        try:
            st.session_state.indexer = OpenSearchIndexerV2()
            
            if not st.session_state.indexer.client.indices.exists(index="rag-documents-v2"):
                st.session_state.indexer.create_index_v2("rag-documents-v2")
            
            st.session_state.retriever = OpenSearchRetriever()
            st.session_state.generator = ResponseGenerator()
            st.session_state.reranker = ClaudeReranker()
            st.session_state.messages = []
            st.session_state.refresh_stats = refresh_stats
            st.session_state.initialized = True
            
            refresh_stats()
            st.session_state.retriever._refresh_available_metadata()
            
        except Exception as e:
            st.error(f"Erreur d'initialisation: {e}")
            st.error("Vérifiez que Docker est lancé avec : docker-compose up -d")
            st.stop()

# ============= INTERFACE =============

# SIDEBAR
with st.sidebar:
    # Upload section
    st.markdown("""
    <div class='section-header'>
        <span class='material-icons'>upload_file</span>
        <h3 style='margin: 0;'>Ajouter un document</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier PDF",
        type=['pdf'],
        help="Taille maximum: 200 MB"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"{uploaded_file.name}")
            st.caption(f"Taille: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        with col2:
            if st.button("Indexer", type="primary", use_container_width=True):
                process_pdf(uploaded_file)
    
    st.divider()
    
    # Métadonnées - uniquement les champs demandés
    if hasattr(st.session_state.retriever, 'available_metadata'):
        with st.expander("Données disponibles", expanded=False):
            metadata = st.session_state.retriever.available_metadata
            
            # Filtrer pour garder uniquement les champs demandés
            fields_to_show = ['filename', 'annee', 'type_document', 'periode']
            
            for field in fields_to_show:
                if field in metadata and metadata[field]:
                    # Formatage du nom du champ
                    field_name = field.replace('_', ' ').title()
                    st.markdown(f"**{field_name}:**")
                    
                    values = metadata[field]
                    # Trier les périodes, sinon ordre normal
                    if field == 'periode':
                        values = sorted(values)
                    
                    for value in values:
                        st.caption(f"• {value}")

# ZONE PRINCIPALE
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='color: #2c3e50; margin-bottom: 10px;'>Seet</h1>
    <p style='color: #7f8c8d;'>Analyse intelligente de documents financiers</p>
</div>
""", unsafe_allow_html=True)

# Tips
with st.expander("Conseils d'utilisation", expanded=False):
    st.markdown("""
    <div class='tips-box'>
        <div class='tips-title'>Comment poser vos questions:</div>
        <ul style='margin: 10px 0; padding-left: 20px;'>
            <li><strong>Périodes:</strong> "premier trimestre 2024", "T1 2024", etc.</li>
            <li><strong>Documents:</strong> Mentionnez "rapport d'activités", "états financiers", etc.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Chat
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("Voir les sources", expanded=False):
                    display_sources(message["sources"])
            
            if "images" in message and message["images"]:
                st.markdown("**Documents visuels:**")
                display_images(message["images"])

# Input
prompt = st.chat_input("Posez votre question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                # Récupérer 15 chunks
                chunks = st.session_state.retriever.hybrid_search(
                    prompt, 
                    k=15,
                    auto_detect_filters=True
                )
                
                # Reranking pour sélectionner les 5 meilleurs
                chunks = st.session_state.reranker.rerank(prompt, chunks, top_k=5)
                
                if not chunks:
                    response_text = "Je n'ai pas trouvé d'informations pertinentes. Vérifiez que le document concerné a été uploadé."
                    st.warning(response_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                else:
                    doc_names = list(set([c.get('filename', 'Document') for c in chunks]))
                    st.info(f"Consultation de: {', '.join(doc_names)}")
                    
                    with st.spinner("Génération de la réponse..."):
                        result = st.session_state.generator.generate_response(
                            prompt, 
                            chunks,
                            include_sources=True,
                            include_images=True
                        )
                    
                    st.markdown(result["response"])
                    
                    if result.get("sources"):
                        with st.expander("Voir les sources", expanded=False):
                            display_sources(result["sources"])
                    
                    if result.get("images"):
                        st.divider()
                        st.markdown("**Documents visuels pertinents:**")
                        display_images(result["images"])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "sources": result.get("sources", []),
                        "images": result.get("images", [])
                    })
                    
            except Exception as e:
                error_msg = f"Erreur: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("OpenSearch V2 & MinIO")
with col2:
    st.caption("Claude 3.7 Sonnet")
with col3:
    st.caption("BGE-M3 Embeddings")