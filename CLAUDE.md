# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Web scraper and analysis tools for CVPR 2024 conference papers (https://openaccess.thecvf.com/CVPR2024). Extracts paper metadata and generates infographic posters with topic clustering.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run scraper
python scraper.py                    # Full scrape (both JSON and CSV)
python scraper.py --limit 10         # Test with 10 papers
python scraper.py --no-abstracts     # Skip fetching individual pages for abstracts

# Generate infographic poster
python poster_generator.py           # Generate poster with auto-clustering (5-7 clusters)
python poster_generator.py --top 3   # Top 3 papers per cluster
python poster_generator.py --clusters 6  # Force 6 clusters
```

## Architecture

- `scraper.py`: Web scraper using requests + BeautifulSoup
  - `fetch_page()`: HTTP fetching with retry logic
  - `extract_paper_links()`: Parses main listing page
  - `extract_abstract()`: Fetches individual paper pages
  - `save_to_json/csv()`: Output formatters

- `poster_generator.py`: Infographic poster generator
  - `generate_embeddings()`: Sentence embeddings via all-MiniLM-L6-v2
  - `find_optimal_clusters()`: K-means with silhouette score optimization
  - `extract_cluster_keywords()`: TF-IDF for cluster naming
  - `rank_papers_in_cluster()`: Ranks by centrality (60%) + author prominence (40%)
  - `generate_html()`: Renders HTML poster with embedded CSS

## Data Structure

Each paper record contains:
- `title`: Paper title
- `authors`: List of author names
- `abstract`: Full abstract text
- `pdf_url`: Link to PDF
- `supplementary_url`: Link to supplementary materials (if available)
- `arxiv_url`: Link to arXiv (if available)
- `paper_page_url`: Link to paper's detail page

## Website Structure

The scraper targets these HTML patterns:
- Paper titles: `<dt class="ptitle">` with nested `<a>` link
- Authors: `<input type="hidden" name="query_author" value="...">` in forms
- Links: `<a>` tags containing "pdf", "supp", "arXiv" text
- Abstracts: `<div id="abstract">` on individual paper pages
