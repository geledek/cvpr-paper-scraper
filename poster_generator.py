"""
CVPR 2024 Infographic Poster Generator

Clusters papers by topic and generates an HTML poster with must-read papers.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from jinja2 import Template


# === Data Loading ===

def load_papers(filepath: str) -> list[dict]:
    """Load papers from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# === Embedding Generation ===

def generate_embeddings(papers: list[dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate sentence embeddings for paper titles and abstracts."""
    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(model_name)

    # Combine title and abstract for richer representation
    texts = []
    for p in papers:
        abstract = p.get("abstract", "") or ""
        texts.append(f"{p['title']}. {abstract}")

    print(f"Generating embeddings for {len(texts)} papers...", flush=True)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


# === Clustering ===

def find_optimal_clusters(embeddings: np.ndarray, k_range: tuple = (5, 8)) -> tuple[int, KMeans, np.ndarray]:
    """Find optimal number of clusters using silhouette score."""
    print(f"Finding optimal clusters in range {k_range}...", flush=True)

    best_k, best_score, best_model, best_labels = k_range[0], -1, None, None

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k}: silhouette score = {score:.4f}", flush=True)

        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, kmeans, labels

    print(f"Optimal k={best_k} with score={best_score:.4f}", flush=True)
    return best_k, best_model, best_labels


def extract_cluster_keywords(papers: list[dict], labels: np.ndarray, n_keywords: int = 8) -> dict[int, list[str]]:
    """Extract distinguishing keywords for each cluster using TF-IDF across clusters."""
    # Combine all papers in each cluster into one document per cluster
    cluster_docs = {}
    unique_labels = sorted(set(labels))

    for paper, label in zip(papers, labels):
        if label not in cluster_docs:
            cluster_docs[label] = []
        abstract = paper.get("abstract", "") or ""
        cluster_docs[label].append(f"{paper['title']} {abstract}")

    # Create one document per cluster
    documents = []
    label_order = []
    for label in unique_labels:
        documents.append(" ".join(cluster_docs[label]))
        label_order.append(label)

    # TF-IDF across all cluster documents to find distinguishing terms
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8  # Ignore terms that appear in >80% of clusters
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}
    for idx, label in enumerate(label_order):
        scores = tfidf_matrix[idx].toarray()[0]
        top_indices = scores.argsort()[-n_keywords:][::-1]
        cluster_keywords[label] = [feature_names[i] for i in top_indices if scores[i] > 0]

    return cluster_keywords


def get_cluster_icon(name: str, keywords: list[str]) -> str:
    """Get an appropriate icon for the cluster based on name and keywords."""
    text = f"{name} {' '.join(keywords)}".lower()

    # Map topics to icons
    if any(x in text for x in ["language", "llm", "caption", "text", "instruction", "mllm"]):
        return "💬"
    if any(x in text for x in ["federated", "distributed", "privacy"]):
        return "🔗"
    if any(x in text for x in ["pose", "skeleton", "body", "human pose"]):
        return "🏃"
    if any(x in text for x in ["diffusion", "generative", "generation", "t2i"]):
        return "🎨"
    if any(x in text for x in ["3d human", "avatar", "mesh", "body"]):
        return "👤"
    if any(x in text for x in ["stereo", "depth", "low light", "hdr", "deblur"]):
        return "📷"
    if any(x in text for x in ["video", "action", "temporal"]):
        return "🎬"
    if any(x in text for x in ["detection", "object"]):
        return "🎯"
    if any(x in text for x in ["segment", "semantic"]):
        return "🧩"
    if any(x in text for x in ["3d", "nerf", "reconstruction", "scene"]):
        return "🌐"
    if any(x in text for x in ["face", "facial"]):
        return "😊"
    if any(x in text for x in ["medical", "clinical"]):
        return "🏥"
    if any(x in text for x in ["autonomous", "driving", "vehicle"]):
        return "🚗"
    if any(x in text for x in ["track", "motion"]):
        return "📍"

    return "🔬"  # Default: research/science icon


def generate_cluster_name(keywords: list[str]) -> str:
    """Generate a human-readable cluster name from keywords."""
    # Map common CV keywords to readable names (ordered by priority)
    keyword_map = [
        ("nerf", "Neural Radiance Fields"),
        ("gaussian", "3D Gaussian Splatting"),
        ("diffusion", "Diffusion Models"),
        ("generative", "Generative Models"),
        ("3d", "3D Vision"),
        ("point cloud", "Point Clouds"),
        ("mesh", "3D Meshes"),
        ("video", "Video Understanding"),
        ("action", "Action Recognition"),
        ("temporal", "Temporal Modeling"),
        ("language", "Vision-Language"),
        ("text", "Text-to-Image"),
        ("llm", "Large Language Models"),
        ("multimodal", "Multimodal Learning"),
        ("detection", "Object Detection"),
        ("segmentation", "Segmentation"),
        ("semantic", "Semantic Understanding"),
        ("instance", "Instance Recognition"),
        ("pose", "Pose Estimation"),
        ("human", "Human Analysis"),
        ("body", "Body Reconstruction"),
        ("face", "Face Analysis"),
        ("hand", "Hand Tracking"),
        ("autonomous", "Autonomous Driving"),
        ("scene", "Scene Understanding"),
        ("depth", "Depth Estimation"),
        ("stereo", "Stereo Vision"),
        ("tracking", "Object Tracking"),
        ("motion", "Motion Analysis"),
        ("flow", "Optical Flow"),
        ("reconstruction", "3D Reconstruction"),
        ("novel view", "Novel View Synthesis"),
        ("rendering", "Neural Rendering"),
        ("image restoration", "Image Restoration"),
        ("super resolution", "Super Resolution"),
        ("denoising", "Image Denoising"),
        ("enhancement", "Image Enhancement"),
        ("editing", "Image Editing"),
        ("inpainting", "Image Inpainting"),
        ("transformer", "Vision Transformers"),
        ("attention", "Attention Mechanisms"),
        ("self supervised", "Self-Supervised Learning"),
        ("contrastive", "Contrastive Learning"),
        ("representation", "Representation Learning"),
        ("domain", "Domain Adaptation"),
        ("adversarial", "Adversarial Learning"),
        ("medical", "Medical Imaging"),
        ("retrieval", "Image Retrieval"),
        ("caption", "Image Captioning"),
        ("vqa", "Visual QA"),
        ("grounding", "Visual Grounding"),
    ]

    keywords_lower = " ".join(keywords).lower()
    matched = []

    for key, name in keyword_map:
        if key in keywords_lower and name not in matched:
            matched.append(name)
            if len(matched) >= 2:
                break

    if matched:
        return " & ".join(matched)

    # Fallback: clean up and use first keyword
    if keywords:
        # Filter out common non-descriptive words
        skip_words = {"model", "models", "method", "methods", "learning", "network", "networks",
                      "based", "using", "novel", "proposed", "approach", "image", "images"}
        clean = [k for k in keywords if k.lower() not in skip_words]
        if clean:
            return clean[0].replace("_", " ").title()
        return keywords[0].replace("_", " ").title()

    return "General Topics"


# === Ranking ===

def compute_author_stats(papers: list[dict]) -> tuple[dict[str, int], int]:
    """Compute author publication frequency."""
    all_authors = []
    for p in papers:
        all_authors.extend(p.get("authors", []))

    author_counts = Counter(all_authors)
    max_count = max(author_counts.values()) if author_counts else 1
    return author_counts, max_count


def rank_papers_in_cluster(
    cluster_papers: list[dict],
    cluster_embeddings: np.ndarray,
    cluster_center: np.ndarray,
    author_counts: dict[str, int],
    max_author_count: int,
    top_n: int = 5,
    centrality_weight: float = 0.6,
    author_weight: float = 0.4
) -> list[dict]:
    """Rank papers by centrality and author prominence."""
    scored_papers = []

    for paper, emb in zip(cluster_papers, cluster_embeddings):
        # Centrality score: cosine similarity to cluster center
        centrality = cosine_similarity([emb], [cluster_center])[0][0]

        # Author prominence score
        authors = paper.get("authors", [])
        if authors:
            total_prominence = sum(author_counts.get(a, 0) for a in authors)
            avg_prominence = total_prominence / len(authors)
            author_score = min(avg_prominence / max_author_count, 1.0)
        else:
            author_score = 0

        # Combined score
        final_score = (centrality_weight * centrality) + (author_weight * author_score)

        scored_papers.append({
            **paper,
            "centrality_score": float(centrality),
            "author_score": float(author_score),
            "final_score": float(final_score)
        })

    # Sort by final score
    scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_papers[:top_n]


# === HTML Generation ===

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CVPR 2024 Must-Read Papers</title>
    <style>
        :root {
            --color-bg: #f5f5f7;
            --color-surface: #ffffff;
            --color-text: #1d1d1f;
            --color-text-secondary: #6e6e73;
            --shadow: 0 2px 12px rgba(0,0,0,0.08);
            --radius: 16px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.5;
        }

        .poster {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .poster-header {
            text-align: center;
            padding: 3rem 1rem;
            margin-bottom: 2rem;
        }

        .poster-header h1 {
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            font-size: 1.5rem;
            color: var(--color-text-secondary);
            margin-bottom: 1rem;
        }

        .stats {
            font-size: 1rem;
            color: var(--color-text-secondary);
            padding: 0.75rem 1.5rem;
            background: var(--color-surface);
            border-radius: 2rem;
            display: inline-block;
            box-shadow: var(--shadow);
        }

        .cluster-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 1.5rem;
        }

        .cluster-section {
            background: var(--color-surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        .cluster-header {
            padding: 1.5rem;
            border-left: 5px solid var(--cluster-color, #007aff);
            background: linear-gradient(90deg, rgba(0,122,255,0.05), transparent);
        }

        .cluster-meta {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .cluster-badge {
            width: 32px;
            height: 32px;
            background: var(--cluster-color, #007aff);
            color: white;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.875rem;
        }

        .cluster-title {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .paper-count {
            font-size: 0.875rem;
            color: var(--color-text-secondary);
        }

        .paper-tiles {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .paper-tile {
            padding: 1rem 1rem 1rem 3rem;
            background: var(--color-bg);
            border-radius: 12px;
            position: relative;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .paper-tile:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .tile-rank {
            position: absolute;
            left: 0.75rem;
            top: 1rem;
            width: 24px;
            height: 24px;
            background: var(--cluster-color, #007aff);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.75rem;
        }

        .paper-title {
            font-size: 0.95rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 0.25rem;
            color: var(--color-text);
        }

        .paper-authors {
            font-size: 0.8rem;
            color: var(--color-text-secondary);
            margin-bottom: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .paper-abstract {
            font-size: 0.8rem;
            color: var(--color-text-secondary);
            line-height: 1.5;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }

        .tile-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
        }

        .score-badge {
            background: rgba(0,122,255,0.1);
            color: #007aff;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 500;
        }

        .paper-links {
            display: flex;
            gap: 0.5rem;
        }

        .paper-links a {
            color: var(--cluster-color, #007aff);
            text-decoration: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: rgba(0,122,255,0.1);
            transition: background 0.2s;
        }

        .paper-links a:hover {
            background: rgba(0,122,255,0.2);
        }

        .poster-footer {
            text-align: center;
            padding: 2rem 1rem;
            color: var(--color-text-secondary);
            font-size: 0.875rem;
        }

        /* Cluster colors */
        .cluster-section:nth-child(1) { --cluster-color: #ff6b6b; }
        .cluster-section:nth-child(2) { --cluster-color: #4ecdc4; }
        .cluster-section:nth-child(3) { --cluster-color: #45b7d1; }
        .cluster-section:nth-child(4) { --cluster-color: #96ceb4; }
        .cluster-section:nth-child(5) { --cluster-color: #dda0dd; }
        .cluster-section:nth-child(6) { --cluster-color: #ffeaa7; }
        .cluster-section:nth-child(7) { --cluster-color: #74b9ff; }

        @media (max-width: 900px) {
            .cluster-grid {
                grid-template-columns: 1fr;
            }
            .poster-header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="poster">
        <header class="poster-header">
            <h1>CVPR 2024</h1>
            <p class="subtitle">Must-Read Papers by Topic</p>
            <p class="stats">{{ total_papers }} papers analyzed &bull; {{ num_clusters }} topic clusters &bull; Top {{ top_n }} per cluster</p>
        </header>

        <main class="cluster-grid">
            {% for cluster in clusters %}
            <section class="cluster-section">
                <div class="cluster-header">
                    <div class="cluster-meta">
                        <span class="cluster-badge">{{ "%02d"|format(cluster.id + 1) }}</span>
                        <h2 class="cluster-title">{{ cluster.name }}</h2>
                    </div>
                    <span class="paper-count">{{ cluster.total_papers }} papers in this cluster</span>
                </div>

                <div class="paper-tiles">
                    {% for paper in cluster.papers %}
                    <article class="paper-tile">
                        <div class="tile-rank">{{ loop.index }}</div>
                        <h3 class="paper-title">{{ paper.title }}</h3>
                        <p class="paper-authors">{{ paper.authors[:4]|join(", ") }}{% if paper.authors|length > 4 %}, et al.{% endif %}</p>
                        <p class="paper-abstract">{{ paper.abstract[:200] }}{% if paper.abstract|length > 200 %}...{% endif %}</p>
                        <div class="tile-footer">
                            <span class="score-badge" title="Centrality: {{ "%.2f"|format(paper.centrality_score) }}, Author: {{ "%.2f"|format(paper.author_score) }}">
                                Score: {{ "%.2f"|format(paper.final_score) }}
                            </span>
                            <div class="paper-links">
                                {% if paper.pdf_url %}<a href="{{ paper.pdf_url }}" target="_blank">PDF</a>{% endif %}
                                {% if paper.arxiv_url %}<a href="{{ paper.arxiv_url }}" target="_blank">arXiv</a>{% endif %}
                            </div>
                        </div>
                    </article>
                    {% endfor %}
                </div>
            </section>
            {% endfor %}
        </main>

        <footer class="poster-footer">
            <p>Data from CVF Open Access &bull; Generated {{ generation_date }} &bull; Clustering via Sentence Transformers</p>
        </footer>
    </div>
</body>
</html>
"""


def generate_html(clusters_data: list[dict], total_papers: int, top_n: int, output_path: str, full_abstract: bool = False):
    """Generate the HTML poster."""
    template = Template(HTML_TEMPLATE)

    html_content = template.render(
        clusters=clusters_data,
        total_papers=total_papers,
        num_clusters=len(clusters_data),
        top_n=top_n,
        generation_date=datetime.now().strftime("%Y-%m-%d"),
        full_abstract=full_abstract
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content


# === One-Page PDF Poster Template ===

ONEPAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CVPR 2024 Must-Read Papers</title>
    <style>
        @page {
            size: A4;
            margin: 12mm;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 8pt;
            line-height: 1.3;
            color: #1a1a2e;
            background: #ffffff;
        }

        .poster {
            width: 100%;
            height: 100%;
        }

        .header {
            text-align: center;
            padding: 6mm 0 4mm 0;
            border-bottom: 2.5px solid #4361ee;
            margin-bottom: 4mm;
        }

        .header h1 {
            font-size: 28pt;
            font-weight: 800;
            color: #1a1a2e;
            letter-spacing: -0.5px;
            margin-bottom: 1mm;
        }

        .header .subtitle {
            font-size: 11pt;
            color: #6c757d;
            font-weight: 400;
        }

        .header .stats {
            font-size: 8pt;
            color: #999;
            margin-top: 2mm;
        }

        .cluster-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4mm;
        }

        .cluster {
            background: #f8f9fa;
            border-radius: 3mm;
            padding: 3mm;
            border-left: 4px solid var(--color);
        }

        .cluster-header {
            display: flex;
            align-items: center;
            margin-bottom: 2mm;
            padding-bottom: 1.5mm;
            border-bottom: 1px solid #e9ecef;
        }

        .cluster-num {
            background: var(--color);
            color: white;
            width: 5mm;
            height: 5mm;
            border-radius: 1mm;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 7pt;
            font-weight: 700;
            margin-right: 2mm;
            flex-shrink: 0;
        }

        .cluster-icon {
            font-size: 14pt;
            margin-right: 2mm;
        }

        .cluster-name {
            font-size: 9pt;
            font-weight: 700;
            color: #1a1a2e;
            flex-grow: 1;
        }

        .cluster-count {
            font-size: 7pt;
            color: #6c757d;
            background: #e9ecef;
            padding: 0.5mm 1.5mm;
            border-radius: 2mm;
        }

        .papers {
            list-style: none;
        }

        .paper {
            margin-bottom: 1.8mm;
            font-size: 7pt;
            line-height: 1.35;
        }

        .paper:last-child {
            margin-bottom: 0;
        }

        .paper-rank {
            color: var(--color);
            font-weight: 700;
            margin-right: 1mm;
            font-size: 6.5pt;
        }

        .paper-title {
            color: #1a1a2e;
            font-weight: 500;
        }

        .paper-authors {
            font-size: 6pt;
            color: #999;
            font-weight: 400;
            margin-left: 1.5mm;
        }

        .footer {
            text-align: center;
            padding-top: 3mm;
            margin-top: 4mm;
            border-top: 1px solid #dee2e6;
            font-size: 7pt;
            color: #999;
        }

        /* Cluster colors */
        .cluster-1 { --color: #e63946; }
        .cluster-2 { --color: #2a9d8f; }
        .cluster-3 { --color: #4361ee; }
        .cluster-4 { --color: #f77f00; }
        .cluster-5 { --color: #7209b7; }
        .cluster-6 { --color: #06d6a0; }
        .cluster-7 { --color: #e07be0; }
    </style>
</head>
<body>
    <div class="poster">
        <header class="header">
            <h1>CVPR 2024</h1>
            <p class="subtitle">Must-Read Papers by Topic</p>
            <p class="stats">{{ total_papers }} papers analyzed across {{ num_clusters }} research clusters</p>
        </header>

        <div class="cluster-grid">
            {% for cluster in clusters %}
            <div class="cluster cluster-{{ loop.index }}">
                <div class="cluster-header">
                    <span class="cluster-num">{{ loop.index }}</span>
                    <span class="cluster-icon">{{ cluster.icon }}</span>
                    <span class="cluster-name">{{ cluster.name }}</span>
                    <span class="cluster-count">{{ cluster.total_papers }}</span>
                </div>
                <ul class="papers">
                    {% for paper in cluster.papers %}
                    <li class="paper">
                        <span class="paper-rank">{{ loop.index }}.</span>
                        <span class="paper-title">{{ paper.title }}</span>
                        <span class="paper-authors">— {{ paper.authors[:3]|join(", ") }}{% if paper.authors|length > 3 %} et al.{% endif %}</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
            {% endfor %}
        </div>

        <footer class="footer">
            Data from CVF Open Access | Generated {{ generation_date }} | Clustering via Sentence Transformers
        </footer>
    </div>
</body>
</html>
"""


def generate_onepage_pdf(clusters_data: list[dict], total_papers: int, output_path: str):
    """Generate a single-page A4 PDF poster."""
    try:
        from weasyprint import HTML
    except ImportError:
        print("ERROR: weasyprint not installed. Install with: pip install weasyprint", flush=True)
        return None

    template = Template(ONEPAGE_TEMPLATE)

    html_content = template.render(
        clusters=clusters_data,
        total_papers=total_papers,
        num_clusters=len(clusters_data),
        generation_date=datetime.now().strftime("%Y-%m-%d")
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    print(f"Rendering one-page PDF...", flush=True)
    HTML(string=html_content).write_pdf(output_path)

    return output_path


# === Multi-Page PDF Template for A4 ===

PDF_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CVPR 2024 Must-Read Papers</title>
    <style>
        @page {
            size: A4;
            margin: 15mm;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 9pt;
            line-height: 1.4;
            color: #1a1a2e;
        }

        .poster-header {
            text-align: center;
            padding: 8mm 0;
            margin-bottom: 5mm;
            border-bottom: 2px solid #4361ee;
        }

        .poster-header h1 {
            font-size: 24pt;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 2mm;
        }

        .subtitle {
            font-size: 14pt;
            color: #6c757d;
            margin-bottom: 2mm;
        }

        .stats {
            font-size: 10pt;
            color: #6c757d;
        }

        .cluster-section {
            margin-bottom: 6mm;
            page-break-inside: avoid;
        }

        .cluster-header {
            background: linear-gradient(90deg, var(--cluster-color, #4361ee) 0%, transparent 100%);
            padding: 3mm 4mm;
            margin-bottom: 3mm;
            border-radius: 2mm;
        }

        .cluster-title {
            font-size: 12pt;
            font-weight: 700;
            color: white;
            display: inline;
        }

        .cluster-badge {
            display: inline-block;
            background: white;
            color: var(--cluster-color, #4361ee);
            width: 6mm;
            height: 6mm;
            line-height: 6mm;
            text-align: center;
            border-radius: 1mm;
            font-weight: 700;
            font-size: 8pt;
            margin-right: 2mm;
        }

        .paper-count {
            font-size: 9pt;
            color: rgba(255,255,255,0.9);
            margin-left: 3mm;
        }

        .paper-tile {
            background: #f8f9fa;
            padding: 3mm;
            margin-bottom: 2mm;
            border-radius: 2mm;
            border-left: 3px solid var(--cluster-color, #4361ee);
            page-break-inside: avoid;
        }

        .paper-header {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1.5mm;
        }

        .tile-rank {
            background: var(--cluster-color, #4361ee);
            color: white;
            width: 5mm;
            height: 5mm;
            line-height: 5mm;
            text-align: center;
            border-radius: 50%;
            font-size: 7pt;
            font-weight: 700;
            margin-right: 2mm;
            flex-shrink: 0;
        }

        .paper-title {
            font-size: 10pt;
            font-weight: 600;
            color: #1a1a2e;
            line-height: 1.3;
        }

        .paper-authors {
            font-size: 8pt;
            color: #6c757d;
            margin-bottom: 1.5mm;
            padding-left: 7mm;
        }

        .paper-abstract {
            font-size: 8pt;
            color: #495057;
            line-height: 1.4;
            text-align: justify;
            padding-left: 7mm;
            margin-bottom: 1.5mm;
        }

        .paper-meta {
            font-size: 7pt;
            color: #6c757d;
            padding-left: 7mm;
        }

        .paper-meta a {
            color: #4361ee;
            text-decoration: none;
            margin-right: 3mm;
        }

        .score-badge {
            background: #e8f4f8;
            color: #4361ee;
            padding: 0.5mm 1.5mm;
            border-radius: 1mm;
            font-size: 7pt;
        }

        .poster-footer {
            text-align: center;
            padding: 4mm 0;
            margin-top: 5mm;
            border-top: 1px solid #dee2e6;
            font-size: 8pt;
            color: #6c757d;
        }

        /* Cluster colors */
        .cluster-1 { --cluster-color: #e63946; }
        .cluster-2 { --cluster-color: #2a9d8f; }
        .cluster-3 { --cluster-color: #4361ee; }
        .cluster-4 { --cluster-color: #f77f00; }
        .cluster-5 { --cluster-color: #7209b7; }
        .cluster-6 { --cluster-color: #06d6a0; }
        .cluster-7 { --cluster-color: #e07be0; }
    </style>
</head>
<body>
    <header class="poster-header">
        <h1>CVPR 2024</h1>
        <p class="subtitle">Must-Read Papers by Topic</p>
        <p class="stats">{{ total_papers }} papers analyzed | {{ num_clusters }} topic clusters | Top {{ top_n }} per cluster</p>
    </header>

    {% for cluster in clusters %}
    <section class="cluster-section cluster-{{ loop.index }}">
        <div class="cluster-header">
            <span class="cluster-badge">{{ "%02d"|format(loop.index) }}</span>
            <h2 class="cluster-title">{{ cluster.name }}</h2>
            <span class="paper-count">({{ cluster.total_papers }} papers)</span>
        </div>

        {% for paper in cluster.papers %}
        <article class="paper-tile">
            <div class="paper-header">
                <span class="tile-rank">{{ loop.index }}</span>
                <h3 class="paper-title">{{ paper.title }}</h3>
            </div>
            <p class="paper-authors">{{ paper.authors|join(", ") }}</p>
            <p class="paper-abstract">{{ paper.abstract or "Abstract not available." }}</p>
            <div class="paper-meta">
                <span class="score-badge">Score: {{ "%.2f"|format(paper.final_score) }}</span>
                {% if paper.pdf_url %}<a href="{{ paper.pdf_url }}">PDF</a>{% endif %}
                {% if paper.arxiv_url %}<a href="{{ paper.arxiv_url }}">arXiv</a>{% endif %}
            </div>
        </article>
        {% endfor %}
    </section>
    {% endfor %}

    <footer class="poster-footer">
        <p>Data from CVF Open Access | Generated {{ generation_date }} | Clustering via Sentence Transformers + K-Means</p>
    </footer>
</body>
</html>
"""


def generate_pdf(clusters_data: list[dict], total_papers: int, top_n: int, output_path: str):
    """Generate A4 PDF poster using weasyprint."""
    try:
        from weasyprint import HTML
    except ImportError:
        print("ERROR: weasyprint not installed. Install with: pip install weasyprint", flush=True)
        return None

    template = Template(PDF_TEMPLATE)

    html_content = template.render(
        clusters=clusters_data,
        total_papers=total_papers,
        num_clusters=len(clusters_data),
        top_n=top_n,
        generation_date=datetime.now().strftime("%Y-%m-%d")
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    print(f"Rendering PDF (this may take a moment)...", flush=True)
    HTML(string=html_content).write_pdf(output_path)

    return output_path


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Generate CVPR 2024 infographic poster")
    parser.add_argument("--input", default="cvpr2024_papers.json", help="Input JSON file")
    parser.add_argument("--output", default="output/cvpr2024_poster", help="Output filename (without extension)")
    parser.add_argument("--clusters", type=int, help="Force number of clusters (default: auto 5-7)")
    parser.add_argument("--top", type=int, default=5, help="Top papers per cluster")
    parser.add_argument("--pdf", action="store_true", help="Generate multi-page A4 PDF with full abstracts")
    parser.add_argument("--onepage", action="store_true", help="Generate single-page A4 PDF poster (topics + titles only)")
    args = parser.parse_args()

    # 1. Load data
    print(f"Loading papers from {args.input}...", flush=True)
    papers = load_papers(args.input)
    print(f"Loaded {len(papers)} papers", flush=True)

    # 2. Generate embeddings
    embeddings = generate_embeddings(papers)

    # 3. Cluster papers
    if args.clusters:
        print(f"Using fixed k={args.clusters}...", flush=True)
        kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        n_clusters = args.clusters
    else:
        n_clusters, kmeans, labels = find_optimal_clusters(embeddings)

    # 4. Extract cluster keywords and names
    print("Extracting cluster keywords...", flush=True)
    cluster_keywords = extract_cluster_keywords(papers, labels)
    cluster_names = {k: generate_cluster_name(v) for k, v in cluster_keywords.items()}

    # 5. Compute author stats
    print("Computing author statistics...", flush=True)
    author_counts, max_author_count = compute_author_stats(papers)

    # 6. Rank papers in each cluster
    print("Ranking papers in each cluster...", flush=True)
    clusters_data = []

    for cluster_id in range(n_clusters):
        # Get papers and embeddings for this cluster
        cluster_mask = labels == cluster_id
        cluster_papers = [p for p, m in zip(papers, cluster_mask) if m]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # Rank papers
        top_papers = rank_papers_in_cluster(
            cluster_papers,
            cluster_embeddings,
            cluster_center,
            author_counts,
            max_author_count,
            top_n=args.top
        )

        clusters_data.append({
            "id": cluster_id,
            "name": cluster_names[cluster_id],
            "icon": get_cluster_icon(cluster_names[cluster_id], cluster_keywords[cluster_id]),
            "keywords": cluster_keywords[cluster_id],
            "total_papers": len(cluster_papers),
            "papers": top_papers
        })

    # Sort clusters by size (largest first)
    clusters_data.sort(key=lambda x: x["total_papers"], reverse=True)

    # 7. Generate output
    output_files = []

    if args.onepage:
        # Generate single-page PDF with top 5 papers
        pdf_path = f"{args.output}_onepage.pdf"
        print(f"Generating single-page A4 PDF poster at {pdf_path}...", flush=True)
        # Limit to top 5 papers for one-page poster
        for cluster in clusters_data:
            cluster["papers"] = cluster["papers"][:5]
        result = generate_onepage_pdf(clusters_data, len(papers), pdf_path)
        if result:
            output_files.append(pdf_path)
    elif args.pdf:
        # Generate multi-page PDF
        pdf_path = f"{args.output}.pdf"
        print(f"Generating A4 PDF poster at {pdf_path}...", flush=True)
        result = generate_pdf(clusters_data, len(papers), args.top, pdf_path)
        if result:
            output_files.append(pdf_path)
    else:
        # Generate HTML
        html_path = f"{args.output}.html"
        print(f"Generating HTML poster at {html_path}...", flush=True)
        generate_html(clusters_data, len(papers), args.top, html_path)
        output_files.append(html_path)

    # Summary
    print("\n" + "="*60, flush=True)
    print("POSTER GENERATION COMPLETE", flush=True)
    print("="*60, flush=True)
    print(f"Total papers: {len(papers)}", flush=True)
    print(f"Clusters: {n_clusters}\n", flush=True)
    for c in clusters_data:
        print(f"  [{c['id']+1:02d}] {c['name']}", flush=True)
        print(f"      Papers: {c['total_papers']}", flush=True)
        print(f"      Keywords: {', '.join(c['keywords'][:5])}", flush=True)
    print(f"\nOutput: {', '.join(output_files)}", flush=True)


if __name__ == "__main__":
    main()
