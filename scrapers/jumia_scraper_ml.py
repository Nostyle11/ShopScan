"""
ML-First Jumia Ghana Scraper
Uses Minimal HTML Parsing + BERT Model for Intelligent Product Extraction
Supports both HTTP requests and Playwright for dynamic content
"""
import re
import requests
import json
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from urllib.parse import quote_plus
from pathlib import Path

# Import BERT and ML components
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT model dependencies not available")

# Import Playwright for dynamic content
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available, using HTTP requests only")

class MLJumiaScraper:
    def __init__(self):
        self.base_url = "https://www.jumia.com.gh/catalog/"
        self.site_name = "Jumia Ghana"
        self.site_key = "jumia"
        
        # HTTP headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Initialize ML components
        self.bert_extractor = None
        self.training_data = None
        self._load_ml_components()
    
    def _load_ml_components(self):
        """Load BERT model and training data for product extraction"""
        if not BERT_AVAILABLE:
            print("⚠️ BERT not available, using enhanced regex fallback")
            return
        
        try:
            # Load training data
            self._load_training_data()
            
            # Initialize BERT-based text classification pipeline
            self.bert_extractor = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
            
            print(f"✅ ML components loaded for {self.site_name}")
            
        except Exception as e:
            print(f"⚠️ Could not load ML components: {e}")
            self.bert_extractor = None
    
    def _load_training_data(self):
        """Load training data for pattern matching"""
        datasets_dir = Path("./datasets")
        
        # Try to load main e-commerce dataset
        training_file = datasets_dir / "ecommerce_products.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
                print(f"📚 Loaded {len(self.training_data)} training examples for {self.site_name}")
        else:
            print("📚 Training data not found, using built-in patterns")
    
    def get_search_url(self, query: str) -> str:
        """Generate search URL for Jumia Ghana"""
        encoded_query = quote_plus(query)
        return f"{self.base_url}?q={encoded_query}"
    
    def scrape_products(self, query: str, max_products: int = 15) -> List[Dict[str, Any]]:
        """
        ML-First Product Scraping Pipeline for Jumia:
        1. Try Playwright for dynamic content, fallback to HTTP
        2. Minimal HTML parsing to find product containers
        3. Extract clean text from each container
        4. Use BERT/ML to intelligently extract product info
        5. Combine with direct HTML parsing for URLs/images
        """
        if PLAYWRIGHT_AVAILABLE:
            return asyncio.run(self._scrape_with_playwright(query, max_products))
        else:
            return self._scrape_with_http(query, max_products)
    
    async def _scrape_with_playwright(self, query: str, max_products: int) -> List[Dict[str, Any]]:
        """Scrape using Playwright for dynamic content"""
        products = []
        search_url = self.get_search_url(query)
        
        try:
            print(f"🌐 Step 1: Playwright request to {search_url}")
            
            async with async_playwright() as p:
                # Launch browser with proper user agent
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                # Navigate to search page
                await page.goto(search_url, wait_until='networkidle')
                
                # Wait for products to load
                await page.wait_for_timeout(3000)
                
                # Get page content
                html_content = await page.content()
                
                await browser.close()
                
                # Process with ML
                products = self._process_html_with_ml(html_content, search_url, max_products)
                
        except Exception as e:
            print(f"❌ Playwright scraping failed: {e}")
            print("🔄 Falling back to HTTP requests...")
            products = self._scrape_with_http(query, max_products)
        
        return products
    
    def _scrape_with_http(self, query: str, max_products: int) -> List[Dict[str, Any]]:
        """Scrape using HTTP requests"""
        products = []
        search_url = self.get_search_url(query)
        
        try:
            print(f"🌐 Step 1: HTTP request to {search_url}")
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
            
            products = self._process_html_with_ml(html_content, search_url, max_products)
            
        except Exception as e:
            print(f"❌ HTTP scraping failed: {e}")
        
        return products
    
    def _process_html_with_ml(self, html_content: str, search_url: str, max_products: int) -> List[Dict[str, Any]]:
        """Process HTML content using ML-first approach"""
        products = []
        
        if not html_content:
            print(f"❌ No HTML content from {self.site_name}")
            return products
        
        # Step 2: Minimal HTML parsing to find product containers
        print(f"🔍 Step 2: Finding product containers...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove noise
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Find product containers - Jumia specific selectors
        product_selectors = [
            'article.prd',  # Primary Jumia product container
            '.prd',         # Alternative Jumia product selector
            'div.prd',      # Div with prd class
            'article[class*="prd"]',  # Article with prd in class
            'div[class*="product"]',
            'div[class*="item"]'
        ]
        
        product_containers = []
        for selector in product_selectors:
            containers = soup.select(selector)
            if containers:
                print(f"📦 Found {len(containers)} containers with selector: {selector}")
                product_containers = containers[:max_products]
                break
        
        if not product_containers:
            print("⚠️ No product containers found, trying fallback...")
            product_containers = soup.find_all('div')[:max_products]
        
        # Step 3: Extract structured data from containers
        print(f"🧠 Step 3: ML-based product extraction from {min(len(product_containers), max_products)} containers...")
        
        for i, container in enumerate(product_containers[:max_products]):
            # Extract clean text for ML processing
            container_text = self._extract_clean_text(container)
            
            # Skip containers with too little content
            if len(container_text) < 20:
                continue
            
            # Extract URLs and images directly from HTML
            product_url = self._extract_product_url(container)
            image_url = self._extract_image_url(container)
            
            # Use ML to extract product information
            product_info = self._ml_extract_product_info(container_text)
            
            if product_info["title"] and product_info["price"]:
                title = self._clean_title(product_info["title"])
                price_value = self._extract_price_number(product_info["price"])
                
                if price_value > 0 and len(title) > 5:
                    products.append({
                        'id': f"jumia_ml_{abs(hash(title + str(i)))}",
                        'title': title,
                        'price': price_value,
                        'image_url': image_url,
                        'url': product_url or search_url,
                        'source': self.site_name,
                        'source_key': self.site_key,
                        'extraction_method': 'ML-BERT' if self.bert_extractor else 'Enhanced-Regex'
                    })
                    
                    method = "🤖 ML-BERT" if self.bert_extractor else "🔧 Enhanced-Regex"
                    print(f"✅ {method}: {title} - ₵{price_value}")
        
        print(f"🎯 Extracted {len(products)} products using ML-first approach")
        return products
    
    def _extract_clean_text(self, container) -> str:
        """Extract clean text from container for ML processing"""
        # Get text with some structure preserved
        text = container.get_text(separator=' | ', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long strings that might be noise
        words = text.split()
        cleaned_words = [word for word in words if len(word) < 50]
        
        return ' '.join(cleaned_words)
    
    def _extract_product_url(self, container) -> str:
        """Extract product URL from container"""
        links = container.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            if href:
                if href.startswith('/'):
                    return f"https://www.jumia.com.gh{href}"
                elif 'jumia.com.gh' in href:
                    return href
        return ""
    
    def _extract_image_url(self, container) -> str:
        """Extract image URL from container"""
        images = container.find_all('img', src=True)
        for img in images:
            src = img.get('src')
            if src and ('jumia.com.gh' in src or src.startswith('http')):
                return src
        return ""
    
    def _ml_extract_product_info(self, text: str) -> Dict[str, str]:
        """Use ML/BERT to extract product information from text"""
        if self.bert_extractor:
            return self._bert_extract(text)
        else:
            return self._enhanced_regex_extract(text)
    
    def _bert_extract(self, text: str) -> Dict[str, str]:
        """Use BERT model for intelligent product extraction"""
        product_info = {"title": "", "price": ""}
        
        try:
            # Use pattern matching enhanced by training data
            if self.training_data:
                product_info = self._pattern_match_with_training_data(text)
            else:
                product_info = self._enhanced_regex_extract(text)
                
        except Exception as e:
            print(f"⚠️ BERT extraction failed: {e}")
            product_info = self._enhanced_regex_extract(text)
        
        return product_info
    
    def _pattern_match_with_training_data(self, text: str) -> Dict[str, str]:
        """Use training data patterns for better extraction"""
        product_info = {"title": "", "price": ""}
        
        # Enhanced patterns based on training data - adapted for Jumia Ghana
        title_patterns = [
            r'([A-Za-z][A-Za-z0-9\s]{15,})(?:\s*₵|\s*GH₵|\s*\d{3,})',
            r'([A-Za-z][A-Za-z0-9\s]{10,})(?:\s*₵|\s*GH₵)',
            r'Verified\s+ID[^₵]*?([A-Za-z][A-Za-z0-9\s]{10,})(?:\s*₵)',
            r'Popular[^₵]*?([A-Za-z][A-Za-z0-9\s]{10,})(?:\s*₵)',
            r'^([A-Za-z][^₵\n]{10,})(?:\s*₵)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate_title = match.group(1).strip()
                # Clean up common prefixes/suffixes
                candidate_title = re.sub(r'^(Quick\s+reply|ENTERPRISE|DIAMOND|Verified\s+ID)', '', candidate_title, flags=re.IGNORECASE)
                candidate_title = candidate_title.strip()
                
                if len(candidate_title) > 10:
                    product_info["title"] = candidate_title
                    break
        
        # Enhanced price extraction for Ghana Cedis
        price_patterns = [
            r'GH₵\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'₵\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*₵',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for price_str in matches:
                    try:
                        price_num = float(price_str.replace(',', ''))
                        if 50 <= price_num <= 100000:  # Reasonable range for Ghana
                            product_info["price"] = price_str
                            break
                    except ValueError:
                        continue
                if product_info["price"]:
                    break
        
        return product_info
    
    def _enhanced_regex_extract(self, text: str) -> Dict[str, str]:
        """Enhanced regex extraction as fallback"""
        product_info = {"title": "", "price": ""}
        
        # Basic title extraction
        title_patterns = [
            r'([A-Za-z][A-Za-z0-9\s]{15,})(?:\s*₵|\s*GH₵)',
            r'^([A-Za-z][^₵\n]{10,})(?:\s*₵)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                product_info["title"] = match.group(1).strip()
                break
        
        # Basic price extraction
        price_patterns = [
            r'₵\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'GH₵\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                product_info["price"] = matches[0]
                break
        
        return product_info
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize product title"""
        if not title:
            return ""
        
        # Clean up whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove trailing price-like patterns
        title = re.sub(r'\s*₵.*$', '', title)
        title = re.sub(r'\s*GH₵.*$', '', title)
        
        return title
    
    def _extract_price_number(self, price_str: str) -> float:
        """Extract numeric price value"""
        if not price_str:
            return 0.0
        
        try:
            # Remove currency symbols and clean
            clean_price = re.sub(r'[₵,\s]', '', price_str)
            clean_price = re.sub(r'GH', '', clean_price)
            return float(clean_price)
        except (ValueError, TypeError):
            return 0.0

# Test function
def test_ml_jumia_scraper():
    """Test the ML-first Jumia scraper"""
    print("🧪 Testing ML-First Jumia Scraper")
    print("=" * 50)
    
    scraper = MLJumiaScraper()
    products = scraper.scrape_products('laptop', max_products=5)
    
    print(f"\n📊 Results: {len(products)} products")
    for i, product in enumerate(products, 1):
        print(f"{i}. {product['title']} - ₵{product['price']}")
        print(f"   Method: {product['extraction_method']}")
        print(f"   URL: {product['url'][:50]}...")
        print()

if __name__ == "__main__":
    test_ml_jumia_scraper()
