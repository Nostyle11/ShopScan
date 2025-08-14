"""
Scraper manager to coordinate all site scrapers
Combines results from all sites into unified product data
Uses ML-First scrapers for intelligent product extraction
"""
from typing import List, Dict, Any
from .ebay_scraper import EbayScraper
from .jiji_scraper_ml import MLJijiScraper
from .jumia_scraper_ml import MLJumiaScraper
from .tonaton_scraper_ml import MLTonatonScraper
from .aliexpress_scraper_ml import MLAliExpressScraper


class ScraperManager:
    def __init__(self):
        self.scrapers = {
            'ebay': EbayScraper(),
            'jiji': MLJijiScraper(),           # 🤖 ML-First
            'jumia': MLJumiaScraper(),         # 🤖 ML-First  
            'tonaton': MLTonatonScraper(),     # 🤖 ML-First
            'aliexpress': MLAliExpressScraper() # 🤖 ML-First
        }
    
    def scrape_all_sites(self, query: str, max_products_per_site: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape products from all sites and combine results
        Returns unified list of products from all sites
        """
        all_products = []
        
        for site_key, scraper in self.scrapers.items():
            try:
                print(f"\n=== Scraping {scraper.site_name} ===")
                products = scraper.scrape_products(query, max_products_per_site)
                all_products.extend(products)
                print(f"✓ Found {len(products)} products from {scraper.site_name}")
            except Exception as e:
                print(f"✗ Error scraping {scraper.site_name}: {str(e)}")
                continue
        
        # Sort by price for better comparison
        all_products.sort(key=lambda x: x.get('price', 0))
        
        print(f"\n🎯 Total products found: {len(all_products)} from {len(self.scrapers)} sites")
        return all_products
    
    def scrape_single_site(self, site_key: str, query: str, max_products: int = 15) -> List[Dict[str, Any]]:
        """
        Scrape products from a specific site
        """
        if site_key not in self.scrapers:
            print(f"Unknown site: {site_key}")
            return []
        
        scraper = self.scrapers[site_key]
        return scraper.scrape_products(query, max_products)
    
    def get_available_sites(self) -> List[str]:
        """
        Get list of available site keys
        """
        return list(self.scrapers.keys())