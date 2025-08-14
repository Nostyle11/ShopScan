"""
Scrapers package - Modular site scrapers for real product data extraction
Each site has its own scraper module following the HTTP + Trafilatura + BERT workflow
"""

from .ebay_scraper import EbayScraper
from .jiji_scraper_ml import MLJijiScraper
from .jumia_scraper_ml import MLJumiaScraper
from .tonaton_scraper_ml import MLTonatonScraper
from .aliexpress_scraper_ml import MLAliExpressScraper

__all__ = [
    'EbayScraper',
    'MLJijiScraper', 
    'MLJumiaScraper',
    'MLTonatonScraper',
    'MLAliExpressScraper'
]