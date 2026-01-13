"""
CVPR 2024 Conference Paper Scraper

Extracts paper titles, authors, abstracts, and links from the CVPR 2024 website.
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
from typing import Optional
from urllib.parse import urljoin

BASE_URL = "https://openaccess.thecvf.com"
LISTING_URL = f"{BASE_URL}/CVPR2024?day=all"


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a page and return parsed BeautifulSoup object."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def extract_paper_links(soup: BeautifulSoup) -> list[dict]:
    """Extract basic paper info and links from the main listing page."""
    papers = []

    # Find all paper title elements
    title_elements = soup.find_all("dt", class_="ptitle")

    for dt in title_elements:
        paper = {}

        # Extract title and paper page URL
        title_link = dt.find("a")
        if title_link:
            paper["title"] = title_link.get_text(strip=True)
            paper["paper_page_url"] = urljoin(BASE_URL, title_link.get("href", ""))
        else:
            continue

        # Get the following dd elements for authors and links
        current = dt.find_next_sibling()
        authors = []

        while current and current.name == "dd":
            # Check for author forms
            author_forms = current.find_all("form", class_="authsearch")
            if author_forms:
                for form in author_forms:
                    author_input = form.find("input", {"name": "query_author"})
                    if author_input:
                        authors.append(author_input.get("value", ""))

            # Check for links (pdf, supp, arXiv)
            links = current.find_all("a")
            for link in links:
                href = link.get("href", "")
                text = link.get_text(strip=True).lower()

                if "pdf" in text and "supp" not in href:
                    paper["pdf_url"] = urljoin(BASE_URL, href) if href.startswith("/") else href
                elif "supp" in text or "supp" in href:
                    paper["supplementary_url"] = urljoin(BASE_URL, href) if href.startswith("/") else href
                elif "arxiv" in text or "arxiv" in href:
                    paper["arxiv_url"] = href

            current = current.find_next_sibling()
            if current and current.name == "dt":
                break

        paper["authors"] = authors
        papers.append(paper)

    return papers


def extract_abstract(paper_url: str) -> Optional[str]:
    """Fetch a paper's page and extract the abstract."""
    soup = fetch_page(paper_url)
    if not soup:
        return None

    abstract_div = soup.find("div", id="abstract")
    if abstract_div:
        return abstract_div.get_text(strip=True)
    return None


def scrape_cvpr2024(limit: Optional[int] = None, include_abstracts: bool = True) -> list[dict]:
    """
    Scrape CVPR 2024 papers.

    Args:
        limit: Maximum number of papers to scrape (None for all)
        include_abstracts: Whether to fetch individual pages for abstracts

    Returns:
        List of paper dictionaries
    """
    print("Fetching main listing page...", flush=True)
    soup = fetch_page(LISTING_URL)
    if not soup:
        print("Failed to fetch listing page", flush=True)
        return []

    print("Extracting paper information...", flush=True)
    papers = extract_paper_links(soup)
    print(f"Found {len(papers)} papers", flush=True)

    if limit:
        papers = papers[:limit]
        print(f"Limiting to {limit} papers", flush=True)

    if include_abstracts:
        print("Fetching abstracts (this may take a while)...", flush=True)
        for i, paper in enumerate(papers):
            if "paper_page_url" in paper:
                abstract = extract_abstract(paper["paper_page_url"])
                paper["abstract"] = abstract

                if (i + 1) % 50 == 0:
                    print(f"Progress: {i + 1}/{len(papers)} papers processed", flush=True)

                # Be polite to the server
                time.sleep(0.3)

    return papers


def save_to_json(papers: list[dict], filename: str = "cvpr2024_papers.json"):
    """Save papers to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(papers)} papers to {filename}")


def save_to_csv(papers: list[dict], filename: str = "cvpr2024_papers.csv"):
    """Save papers to CSV file."""
    if not papers:
        return

    # Define CSV columns
    fieldnames = ["title", "authors", "abstract", "pdf_url", "supplementary_url", "arxiv_url", "paper_page_url"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for paper in papers:
            row = paper.copy()
            # Convert authors list to semicolon-separated string
            row["authors"] = "; ".join(paper.get("authors", []))
            writer.writerow(row)

    print(f"Saved {len(papers)} papers to {filename}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape CVPR 2024 conference papers")
    parser.add_argument("--limit", type=int, help="Limit number of papers to scrape")
    parser.add_argument("--no-abstracts", action="store_true", help="Skip fetching abstracts")
    parser.add_argument("--output", default="cvpr2024_papers", help="Output filename (without extension)")
    parser.add_argument("--format", choices=["json", "csv", "both"], default="both", help="Output format")

    args = parser.parse_args()

    papers = scrape_cvpr2024(
        limit=args.limit,
        include_abstracts=not args.no_abstracts
    )

    if not papers:
        print("No papers scraped")
        return

    if args.format in ["json", "both"]:
        save_to_json(papers, f"{args.output}.json")

    if args.format in ["csv", "both"]:
        save_to_csv(papers, f"{args.output}.csv")

    print(f"\nScraping complete! Total papers: {len(papers)}")


if __name__ == "__main__":
    main()
