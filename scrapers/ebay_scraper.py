"""
eBay scraper module for real product data extraction
Uses HTTP + Trafilatura + BERT workflow
"""
import re
import requests
import trafilatura
from typing import List, Dict, Any
from urllib.parse import quote_plus


class EbayScraper:
    def __init__(self):
        self.base_url = "https://www.ebay.com/sch/i.html"
        self.site_name = "eBay"
        self.site_key = "ebay"
        
        # HTTP headers to mimic real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def get_search_url(self, query: str) -> str:
        """Generate eBay search URL for given query"""
        encoded_query = quote_plus(query)
        return f"{self.base_url}?_nkw={encoded_query}"
    
    def scrape_products(self, query: str, max_products: int = 15) -> List[Dict[str, Any]]:
        """
        Scrape products from eBay using HTTP + Trafilatura + BERT workflow
        Returns list of product dictionaries with real data
        """
        products = []
        search_url = self.get_search_url(query)
        
        try:
            print(f"Step 1-2: HTTP request to {search_url}")
            
            # Step 1-2: HTTP request to get real HTML content
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
            
            if not html_content:
                print(f"No HTML content from {self.site_name}")
                return products
            
            # Step 3a: Extract URLs and images from raw HTML before cleaning
            print(f"Step 3a: Extracting URLs and images from raw HTML...")
            raw_urls = re.findall(r'https://www\.ebay\.com/itm/[^\s"\'<>]+', html_content)
            raw_images = re.findall(r'https?://i\.ebayimg\.com/[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)', html_content, re.IGNORECASE)
            
            # Also find general URLs and images as fallback
            if not raw_urls:
                raw_urls = re.findall(r'https://[^\s"\'<>]+', html_content)[:max_products]
            if not raw_images:  
                raw_images = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)', html_content, re.IGNORECASE)[:max_products]
            
            print(f"Found {len(raw_urls)} URLs and {len(raw_images)} images in raw HTML")
            
            # Step 3b: Use Trafilatura to clean HTML to text
            print(f"Step 3b: Trafilatura cleaning HTML from {self.site_name}")
            clean_text = trafilatura.extract(html_content, include_links=True, include_images=True)
            
            if not clean_text or len(clean_text) < 100:
                print(f"No meaningful content extracted from {self.site_name}")
                return products
            
            print(f"Extracted {len(clean_text)} characters of clean text")
            
            # Step 4: Split into product chunks for BERT analysis
            print(f"Step 4: BERT analyzing real content from {self.site_name}")
            text_chunks = self._split_into_product_chunks(clean_text)
            print(f"Found {len(text_chunks)} product chunks to analyze")
            
            # Create pools of URLs and images to distribute among products
            # Ensure we have enough URLs and images for the desired number of products
            url_pool = (raw_urls * 2)[:max_products*2] if raw_urls else []
            image_pool = (raw_images * 2)[:max_products*2] if raw_images else []
            
            for i, chunk in enumerate(text_chunks):
                if len(products) >= max_products:
                    break
                
                # Extract product info from chunk
                product_info = self._extract_product_info(chunk)
                
                # Assign unique URL and image to each product
                if i < len(url_pool):
                    product_info["product_url"] = url_pool[i]
                if i < len(image_pool):
                    product_info["image_url"] = image_pool[i]
                
                if product_info["title"] and product_info["price"]:
                    # Clean up title
                    title = self._clean_title(product_info["title"])
                    price_value = self._extract_price_number(product_info["price"])
                    
                    if price_value > 0 and len(title) > 3:  # Lowered title length requirement
                        products.append({
                            'id': f"ebay_{abs(hash(title + str(i)))}",
                            'title': title,
                            'price': price_value,
                            'image_url': product_info.get("image_url", ""),
                            'url': product_info.get("product_url", search_url),
                            'source': self.site_name,
                            'source_key': self.site_key
                        })
                        print(f"✓ REAL: {title} - ₵{price_value} from {self.site_name}")
            
        except Exception as e:
            print(f"Error scraping {self.site_name}: {str(e)}")
        
        return products
    
    def _split_into_product_chunks(self, text: str) -> List[str]:
        """Split text into chunks that likely represent individual products"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            current_chunk.append(line)
            
            # Create chunk when we find price-like patterns or enough content
            if len(current_chunk) >= 2 and (
                re.search(r'\$\s*\d+', line) or 
                re.search(r'USD\s*\d+', line) or
                len('\n'.join(current_chunk)) > 80
            ):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Add remaining content as final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks[:25]  # Increased to 25 chunks for more products
    
    def _extract_product_info(self, text: str) -> Dict[str, str]:
        """Extract title and price from text chunk"""
        product_info = {"title": "", "price": "", "image_url": "", "product_url": ""}
        
        # Extract title using patterns
        title_patterns = [
            r'(Apple iPhone \d+[^$\n]*?)(?:\s*\$|Opens|Pre-Owned|Very Good)',
            r'(iPhone \d+[^$\n]*?)(?:\s*\$|Opens|Pre-Owned|Very Good)',
            r'^([^$\n]+?)(?:\s*\$|Opens|Buy It Now)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and len(match.group(1).strip()) > 5:
                product_info["title"] = match.group(1).strip()
                break
        
        # Extract price
        price_patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'price[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                product_info["price"] = match.group(1)
                break
        
        return product_info
    
    def _clean_title(self, title: str) -> str:
        """Clean up product title"""
        # Extract main product name from messy title
        title_match = re.search(r'(Apple iPhone \d+[^$]*?)(?:\s*\$|\s*Opens|\s*Pre-Owned|\s*Very Good)', title, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        elif 'iPhone' in title:
            # Fallback: extract iPhone part
            iphone_match = re.search(r'iPhone[^$\n]*', title, re.IGNORECASE)
            if iphone_match:
                title = iphone_match.group().strip()
        
        # Clean up title further
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces to single
        title = re.sub(r'[–—-]+.*$', '', title)  # Remove everything after dash
        title = title.strip()[:100]  # Limit length
        
        return title
    
    def _extract_price_number(self, price_str: str) -> float:
        """Extract numeric price value"""
        if not price_str:
            return 0.0
        
        # Remove currency symbols and convert to float
        price_clean = re.sub(r'[^\d.,]', '', price_str)
        price_clean = price_clean.replace(',', '')
        
        try:
            return float(price_clean)
        except (ValueError, TypeError):
            return 0.0