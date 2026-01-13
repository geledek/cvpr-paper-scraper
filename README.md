# CVPR 2024 Paper Scraper & Poster Generator

Extract paper metadata from the CVPR 2024 conference and generate infographic posters with automatic topic clustering.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Web Scraper**: Extract titles, authors, abstracts, and PDF links from 2,700+ CVPR 2024 papers
- **Topic Clustering**: Automatic clustering using sentence embeddings + K-means
- **Poster Generator**: Create infographic posters (HTML/PDF) with must-read papers per topic
- **Ranking**: Papers ranked by cluster centrality and author prominence

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

### 1. Scrape Papers

```bash
# Full scrape (takes ~30 minutes)
uv run python scraper.py

# Test with limited papers
uv run python scraper.py --limit 10
```

Output: `cvpr2024_papers.json` and `cvpr2024_papers.csv`

### 2. Generate Poster

```bash
# One-page PDF infographic (recommended)
uv run python poster_generator.py --onepage

# Interactive HTML version
uv run python poster_generator.py

# Multi-page PDF with full abstracts
uv run python poster_generator.py --pdf
```

Output: `output/cvpr2024_poster_onepage.pdf`

## Sample Output

The one-page poster includes:
- 6 automatically discovered topic clusters
- Top 5 must-read papers per cluster
- Author names and paper counts
- Color-coded sections with icons

**Topics discovered from CVPR 2024:**
| Cluster | Topic | Papers |
|---------|-------|--------|
| 1 | Large Language Models & Vision | 648 |
| 2 | Federated Learning | 504 |
| 3 | Pose Estimation & Scene Understanding | 467 |
| 4 | Diffusion Models & 3D Vision | 401 |
| 5 | Image Enhancement & Stereo | 380 |
| 6 | 3D Human & Avatar | 316 |

## How It Works

1. **Scraping**: Fetches paper metadata from [CVF Open Access](https://openaccess.thecvf.com/CVPR2024)
2. **Embedding**: Generates sentence embeddings using `all-MiniLM-L6-v2`
3. **Clustering**: K-means clustering with silhouette score optimization
4. **Ranking**: Combined score of cluster centrality (60%) + author prominence (40%)
5. **Rendering**: HTML/CSS to PDF via WeasyPrint

## Project Structure

```
├── scraper.py           # Web scraper for CVPR papers
├── poster_generator.py  # Clustering and poster generation
├── pyproject.toml       # Project dependencies (uv)
├── requirements.txt     # Dependencies (pip fallback)
└── output/              # Generated posters
```

## License

MIT
