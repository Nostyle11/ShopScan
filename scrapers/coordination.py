"""
Coordination module for product scraping
This replaces the old scraper.py with a cleaner approach
"""
from typing import List, Dict, Any
from .scraper_manager import ScraperManager


def scrape_products(query: str, max_products: int = 20) -> List[Dict[str, Any]]:
    """
    Main entry point for product scraping using modular site scrapers
    Uses the exact 5-step workflow: HTTP → Extract URLs → Trafilatura → BERT → JSON
    
    Returns only REAL scraped data from actual websites - no fallbacks or fake data
    """
    manager = ScraperManager()
    
    # Calculate products per site to reach total max_products
    products_per_site = max(1, max_products // len(manager.get_available_sites()))
    
    # Scrape from all sites using modular scrapers
    all_products = manager.scrape_all_sites(query, products_per_site)
    
    # Limit to max_products and return
    return all_products[:max_products]