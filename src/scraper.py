"""
Web scraper for additional data sources.
Fetches content from Pittsburgh/CMU related websites.
"""
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WebScraper:
    """Scraper for Pittsburgh/CMU websites."""

    def __init__(self, output_dir: str = "scraped_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.scraped_urls = set()

    def fetch_page(self, url: str, verify_ssl: bool = True) -> Optional[str]:
        """Fetch a webpage and return its HTML content."""
        if url in self.scraped_urls:
            return None

        try:
            response = self.session.get(url, timeout=15, verify=verify_ssl)
            response.raise_for_status()
            self.scraped_urls.add(url)
            time.sleep(0.5) 
            return response.text
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None

    def extract_text(self, html: str, url: str) -> Dict:
        """Extract text content from HTML."""
        soup = BeautifulSoup(html, 'lxml')

        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            tag.decompose()

        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else urlparse(url).path

        # Get main content
        main = soup.find('main') or soup.find('article') or soup.find('body')
        if main:
            text = main.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        return {
            'url': url,
            'title': title_text,
            'text': text
        }

    def save_document(self, doc: Dict, filename: str):
        """Save document to file."""
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"  Saved: {filepath}")

    def scrape_url(self, url: str, name: str, verify_ssl: bool = True) -> Optional[Dict]:
        """Scrape a single URL and save it."""
        print(f"Scraping: {url}")
        html = self.fetch_page(url, verify_ssl)
        if html:
            doc = self.extract_text(html, url)
            if len(doc['text']) > 100:
                self.save_document(doc, name)
                return doc
        return None

    def scrape_with_subpages(self, base_url: str, name: str, max_pages: int = 10, verify_ssl: bool = True) -> List[Dict]:
        """Scrape a page and its linked subpages."""
        docs = []

        print(f"Scraping: {base_url}")
        html = self.fetch_page(base_url, verify_ssl)
        if not html:
            return docs

        doc = self.extract_text(html, base_url)
        if len(doc['text']) > 100:
            self.save_document(doc, f"{name}_main")
            docs.append(doc)

        soup = BeautifulSoup(html, 'lxml')
        base_domain = urlparse(base_url).netloc

        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            if parsed.netloc == base_domain and full_url not in self.scraped_urls:
                if not any(ext in parsed.path.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '#']):
                    links.append(full_url)

        # Scrape subpages
        links = list(set(links))[:max_pages]
        for i, link in enumerate(links):
            html = self.fetch_page(link, verify_ssl)
            if html:
                doc = self.extract_text(html, link)
                if len(doc['text']) > 100:
                    self.save_document(doc, f"{name}_sub_{i}")
                    docs.append(doc)

        return docs


def scrape_all_sources():
    """Scrape all recommended data sources."""
    scraper = WebScraper("scraped_data")

    # CMU Pages
    cmu_pages = [
        ("https://www.cmu.edu/about/", "cmu_about"),
        ("https://www.cmu.edu/about/history.html", "cmu_history"),
        ("https://www.cmu.edu/about/rankings.html", "cmu_rankings"),
        ("https://www.cmu.edu/engage/alumni/events/campus/index.html", "cmu_campus_events"),
    ]
    for url, name in cmu_pages:
        scraper.scrape_url(url, name)

    # Pittsburgh Info
    pgh_pages = [
        ("https://www.britannica.com/place/Pittsburgh", "britannica_pittsburgh"),
        ("https://www.visitpittsburgh.com/about-pittsburgh/", "visitpgh_about"),
        ("https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/", "visitpgh_sports"),
        ("https://www.visitpittsburgh.com/events-festivals/food-festivals/", "visitpgh_food_festivals"),
    ]
    for url, name in pgh_pages:
        scraper.scrape_url(url, name)

    # Sports
    sports_pages = [
        ("https://www.steelers.com/team/history/", "steelers_history"),
        ("https://www.mlb.com/pirates/history", "pirates_history"),
        ("https://www.nhl.com/penguins/team/history", "penguins_history"),
    ]
    for url, name in sports_pages:
        scraper.scrape_url(url, name, verify_ssl=False)

    # Food Festivals
    food_pages = [
        ("https://www.picklesburgh.com/", "picklesburgh"),
        ("https://www.pghtacofest.com/", "taco_fest"),
        ("https://pittsburghrestaurantweek.com/", "restaurant_week"),
        ("https://littleitalydays.com/", "little_italy_days"),
    ]
    for url, name in food_pages:
        scraper.scrape_url(url, name, verify_ssl=False)

    # Museums/Culture 
    culture_pages = [
        ("https://www.pittsburghsymphony.org/about", "pgh_symphony"),
        ("https://pittsburghopera.org/about/", "pgh_opera"),
        ("https://carnegiemuseums.org/about/", "carnegie_museums"),
        ("https://www.heinzhistorycenter.org/about/", "heinz_history"),
        ("https://www.thefrickpittsburgh.org/About", "the_frick"),
    ]
    for url, name in culture_pages:
        scraper.scrape_url(url, name, verify_ssl=False)

    # Events 
    event_pages = [
        ("https://downtownpittsburgh.com/events/", "downtown_events"),
    ]
    for url, name in event_pages:
        scraper.scrape_url(url, name, verify_ssl=False)

    print(f"Output directory: {scraper.output_dir}")


def convert_scraped_to_html(scraped_dir: str = "scraped_data", output_dir: str = "additional_data"):
    """Convert scraped JSON files to simple HTML for processing."""
    scraped_path = Path(scraped_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_file in scraped_path.glob("*.json"):
        with open(json_file, 'r') as f:
            doc = json.load(f)

        # Create simple HTML
        html = f"""<!DOCTYPE html>
<html>
<head><title>{doc['title']}</title></head>
<body>
<h1>{doc['title']}</h1>
<p>Source: {doc['url']}</p>
<div>{doc['text'].replace(chr(10), '<br>')}</div>
</body>
</html>"""

        output_file = output_path / f"{json_file.stem}.html"
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Converted: {output_file}")


if __name__ == "__main__":
    scrape_all_sources()
    convert_scraped_to_html()
