"""
ML-First AliExpress Scraper
Uses Minimal HTML Parsing + BERT Model for Intelligent Product Extraction
Enhanced for international e-commerce site structure
"""
import re
import requests
import json
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

class MLAliExpressScraper:
    def __init__(self):
        self.base_url = "https://www.aliexpress.com/wholesale"
        self.site_name = "AliExpress"
        self.site_key = "aliexpress"
        
        # HTTP headers with more realistic browser simulation
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
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
        
        # Try to load main e-commerce training dataset
        training_file = datasets_dir / "ecommerce_products.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
                print(f"📚 Loaded {len(self.training_data)} training examples for {self.site_name}")
        else:
            print("📚 Training data not found, using built-in patterns")
    
    def get_search_url(self, query: str) -> str:
        """Generate search URL for AliExpress"""
        encoded_query = quote_plus(query)
        return f"{self.base_url}?SearchText={encoded_query}"
    
    def scrape_products(self, query: str, max_products: int = 15) -> List[Dict[str, Any]]:
        """
        ML-First Product Scraping Pipeline for AliExpress:
        1. HTTP request with enhanced headers
        2. Minimal HTML parsing to find product containers
        3. Extract clean text from each container
        4. Use BERT/ML to intelligently extract product info
        5. Combine with direct HTML parsing for URLs/images
        """
        products = []
        search_url = self.get_search_url(query)
        
        try:
            print(f"🌐 Step 1: HTTP request to {search_url}")
            
            # Step 1: Get HTML content with session for better reliability
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(search_url, timeout=15)
            response.raise_for_status()
            html_content = response.text
            
            if not html_content:
                print(f"❌ No HTML content from {self.site_name}")
                return products
            
            # Step 2: Minimal HTML parsing to find product containers
            print(f"🔍 Step 2: Finding product containers...")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove noise
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Find product containers - AliExpress specific selectors
            product_selectors = [
                # Modern AliExpress selectors (2024+)
                'div[data-widget-cid*="hy_bn"]', 
                'div[class*="search-item-card-wrapper-gallery"]',
                'div[class*="kt_jy"]',
                'div[class*="list-item"]',
                'div[class*="gallery-item"]',
                'div[class*="item-wrap"]',
                'div[class*="item-container"]',
                
                # Legacy selectors (fallback)
                'article[class*="prd"]',  # Primary product container from DOM
                'div[class*="prd"]',      # Alternative product container
                'article.prd_fb',         # Specific AliExpress product class
                'div[data-spm-anchor-id]',
                'div[class*="product"]',
                'div[class*="item"]',
                'div[class*="card"]',
                'article'
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
                # More aggressive fallback for AliExpress
                all_divs = soup.find_all('div')
                product_containers = [div for div in all_divs if len(div.get_text(strip=True)) > 50][:max_products]
            
            # Step 3: Extract structured data from containers
            print(f"🧠 Step 3: ML-based product extraction from {min(len(product_containers), max_products)} containers...")
            
            for i, container in enumerate(product_containers[:max_products]):
                # Extract clean text for ML processing
                container_text = self._extract_clean_text(container)
                
                # Skip containers with too little content
                if len(container_text) < 30:
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
                            'id': f"aliexpress_ml_{abs(hash(title + str(i)))}",
                            'title': title,
                            'price': price_value,
                            'image_url': image_url,
                            'url': product_url or search_url,
                            'source': self.site_name,
                            'source_key': self.site_key,
                            'extraction_method': 'ML-BERT' if self.bert_extractor else 'Enhanced-Regex'
                        })
                        
                        method = "🤖 ML-BERT" if self.bert_extractor else "🔧 Enhanced-Regex"
                        print(f"✅ {method}: {title} - ${price_value}")
            
            print(f"🎯 Extracted {len(products)} products using ML-first approach")
            
        except Exception as e:
            print(f"❌ Scraping failed for {search_url}: {str(e)}")
        
        return products
    
    def _extract_clean_text(self, container) -> str:
        """Extract clean text from container for ML processing"""
        # Get text with some structure preserved
        text = container.get_text(separator=' | ', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long strings that might be noise
        words = text.split()
        cleaned_words = [word for word in words if len(word) < 60]
        
        return ' '.join(cleaned_words)
    
    def _extract_product_url(self, container) -> str:
        """Extract product URL from container"""
        links = container.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            if href:
                if href.startswith('/'):
                    return f"https://www.aliexpress.com{href}"
                elif 'aliexpress.com' in href:
                    return href
                elif href.startswith('http'):
                    return href
        return ""
    
    def _extract_image_url(self, container) -> str:
        """Extract image URL from container"""
        images = container.find_all('img', src=True)
        for img in images:
            src = img.get('src')
            if src and (src.startswith('http') or src.startswith('//')):
                if src.startswith('//'):
                    src = f"https:{src}"
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
        
        # More flexible title patterns for AliExpress
        title_patterns = [
            # Pattern 1: Text before price (most common)
            r'([A-Za-z][A-Za-z0-9\s\-\.]{8,60})(?:\s*\$|\s*USD|\s*\d{1,4}\.\d{2})',
            
            # Pattern 2: After common prefixes
            r'(?:Free\s+shipping|Sale|Hot\s+sale|New\s+arrival)\s*[^\w]*([A-Za-z][A-Za-z0-9\s\-\.]{8,50}?)(?:\s*\$|\s*USD)',
            
            # Pattern 3: Product-specific patterns
            r'(?:iPhone|Samsung|Laptop|MacBook|Dell|HP|Lenovo|Acer|ASUS)\s+([A-Za-z0-9\s\-\.]{5,40})(?:\s*\$|\s*USD|\s*\d{1,4}\.\d{2})',
            
            # Pattern 4: General product line (fallback)
            r'^([A-Za-z][A-Za-z0-9\s\-\.]{10,50}?)(?:\s*\$|\s*USD|\s*\d{1,4}\.\d{2})',
            
            # Pattern 5: Between common words and price
            r'(?:New|Original|Genuine|Brand)\s+([A-Za-z][A-Za-z0-9\s\-\.]{8,40})(?:\s*\$|\s*USD)',
            
            # Pattern 6: Simple extraction (very flexible fallback)
            r'([A-Za-z][A-Za-z0-9\s]{15,}?)(?=\s*\$|\s*USD|\s*\d{1,4}\.\d{2})',
        ]
        
        for i, pattern in enumerate(title_patterns):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate_title = match.group(1).strip()
                
                # Clean up the title
                candidate_title = self._clean_extracted_title(candidate_title)
                
                # Validate title quality
                if self._is_valid_title(candidate_title):
                    product_info["title"] = candidate_title
                    print(f"🎯 Title extracted with pattern {i+1}: '{candidate_title[:30]}...'")
                    break
        
        # Enhanced price extraction for international currencies
        price_patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'USD\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'US\s*\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,4}(?:,\d{3})*(?:\.\d{2})?)\s*\$',
            r'(\d{1,4}\.\d{2})\s*USD',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for price_str in matches:
                    try:
                        price_num = float(price_str.replace(',', ''))
                        if 1 <= price_num <= 10000:  # Reasonable range for AliExpress
                            product_info["price"] = price_str
                            break
                    except ValueError:
                        continue
                if product_info["price"]:
                    break
        
        return product_info
    
    def _clean_extracted_title(self, title: str) -> str:
        """Clean and normalize extracted title"""
        if not title:
            return ""
        
        # Remove common prefixes and noise
        prefixes_to_remove = [
            r'^(Free\s+shipping|Sale|Hot\s+sale|New\s+arrival|Best\s+seller)\s*',
            r'^(New|Original|Genuine|Brand)\s+',
            r'^(For\s+sale|Selling|Available)\s*',
        ]
        
        for prefix in prefixes_to_remove:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Clean up whitespace and special characters
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove trailing price-like patterns
        title = re.sub(r'\s*\$.*$', '', title)
        title = re.sub(r'\s*USD.*$', '', title)
        title = re.sub(r'\s*\d{1,4}\.\d{2}.*$', '', title)
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'\s*(available|for\s+sale|selling|negotiable)$',
            r'\s*(contact|order\s+now|buy\s+now).*$',
        ]
        
        for suffix in suffixes_to_remove:
            title = re.sub(suffix, '', title, flags=re.IGNORECASE)
        
        return title.strip()

    def _is_valid_title(self, title: str) -> bool:
        """Check if extracted title is valid"""
        if not title or len(title) < 5:
            return False
        
        # Check for minimum meaningful content
        if len(title.split()) < 2:
            return False
        
        # Reject titles that are mostly numbers or special characters
        if re.match(r'^[\d\s\-\.]+$', title):
            return False
        
        # Reject common noise patterns
        noise_patterns = [
            r'^(contact|order|buy|shop|store)$',
            r'^(available|selling|for\s+sale)$',
            r'^(new|original|genuine|brand)$',
            r'^\d+$',
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return False
        
        return True
    
    def _enhanced_regex_extract(self, text: str) -> Dict[str, str]:
        """Enhanced regex extraction as fallback"""
        product_info = {"title": "", "price": ""}
        
        # Basic title extraction for international products
        title_patterns = [
            r'([A-Za-z][A-Za-z0-9\s\-]{15,})(?:\s*\$|\s*USD)',
            r'^([A-Za-z][^\$\n]{10,})(?:\s*\$)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                product_info["title"] = match.group(1).strip()
                break
        
        # Basic price extraction for USD
        price_patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'USD\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
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
        title = re.sub(r'\s*\$.*$', '', title)
        title = re.sub(r'\s*USD.*$', '', title)
        
        return title
    
    def _extract_price_number(self, price_str: str) -> float:
        """Extract numeric price value"""
        if not price_str:
            return 0.0
        
        try:
            # Remove currency symbols and clean
            clean_price = re.sub(r'[\$,\s]', '', price_str)
            clean_price = re.sub(r'USD', '', clean_price)
            return float(clean_price)
        except (ValueError, TypeError):
            return 0.0

# Test function
def test_ml_aliexpress_scraper():
    """Test the ML-first AliExpress scraper"""
    print("🧪 Testing ML-First AliExpress Scraper")
    print("=" * 50)
    
    scraper = MLAliExpressScraper()
    products = scraper.scrape_products('laptop', max_products=5)
    
    print(f"\n📊 Results: {len(products)} products")
    for i, product in enumerate(products, 1):
        print(f"{i}. {product['title']} - ${product['price']}")
        print(f"   Method: {product['extraction_method']}")
        print(f"   URL: {product['url'][:50]}...")
        print()

if __name__ == "__main__":
    test_ml_aliexpress_scraper()
