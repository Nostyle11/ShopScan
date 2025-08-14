"""
BERT-Powered Intelligent Web Scraper
Replaces hard-coded HTML patterns with machine learning-based content extraction
"""

import re
import json
import logging
import requests
import trafilatura
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline,
    AutoModel
)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

logger = logging.getLogger(__name__)

class BERTScraper:
    """
    Intelligent web scraper powered by BERT models
    Learns patterns instead of using hard-coded rules
    """
    
    def __init__(self, model_dir: str = "finetuned-bert-product"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT models
        self.tokenizer = None
        self.ner_model = None
        self.similarity_model = None
        self.ner_pipeline = None
        
        # Content analysis components
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3)
        )
        
        # Common headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Load or initialize models
        self.load_models()
        
        logger.info("BERT Scraper initialized")
    
    def load_models(self):
        """Load BERT models for content analysis"""
        try:
            # Try to load fine-tuned model first
            if os.path.exists(self.model_dir):
                logger.info(f"Loading fine-tuned model from {self.model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.ner_model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.ner_model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                logger.info("Loading pre-trained BERT model")
                model_name = "bert-base-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.ner_model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=9  # O, B-TITLE, I-TITLE, B-PRICE, I-PRICE, B-URL, I-URL, B-IMG, I-IMG
                )
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.ner_model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Load similarity model for content relevance
            self.similarity_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic models
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.similarity_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def scrape_products(self, query: str, urls: List[str] = None, max_products: int = 15) -> List[Dict[str, Any]]:
        """
        Intelligently scrape products using BERT-powered analysis
        
        Args:
            query: Search query for products
            urls: Optional list of URLs to scrape (if None, will search common sites)
            max_products: Maximum number of products to return
        
        Returns:
            List of product dictionaries
        """
        products = []
        
        # If no URLs provided, generate search URLs for common sites
        if not urls:
            urls = self._generate_search_urls(query)
        
        for url in urls:
            try:
                site_products = self._scrape_single_site(url, query, max_products - len(products))
                products.extend(site_products)
                
                if len(products) >= max_products:
                    break
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        # Post-process and rank products
        products = self._rank_and_filter_products(products, query)
        
        return products[:max_products]
    
    def _generate_search_urls(self, query: str) -> List[str]:
        """Generate search URLs for common e-commerce sites"""
        encoded_query = quote_plus(query)
        
        search_urls = [
            f"https://www.jumia.com.gh/catalog/?q={encoded_query}",
            f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}",
            f"https://www.aliexpress.com/wholesale?SearchText={encoded_query}",
            f"https://jiji.com.gh/search?query={encoded_query}",
            f"https://tonaton.com/s_{encoded_query}"
        ]
        
        return search_urls
    
    def _scrape_single_site(self, url: str, query: str, max_products: int) -> List[Dict[str, Any]]:
        """Scrape a single website using BERT analysis"""
        products = []
        
        try:
            # Step 1: Fetch HTML content
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            html_content = response.text
            
            if not html_content:
                return products
            
            # Step 2: Extract clean text using Trafilatura
            clean_text = trafilatura.extract(
                html_content, 
                include_links=True, 
                include_images=True,
                include_tables=True
            )
            
            if not clean_text or len(clean_text) < 100:
                return products
            
            # Step 3: Extract URLs and images from raw HTML
            raw_urls = self._extract_urls_from_html(html_content, url)
            raw_images = self._extract_images_from_html(html_content)
            
            # Step 4: Use BERT to analyze content and extract products
            products = self._bert_extract_products(
                clean_text, 
                query, 
                raw_urls, 
                raw_images, 
                url,
                max_products
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return products
    
    def _bert_extract_products(self, text: str, query: str, urls: List[str], 
                              images: List[str], base_url: str, max_products: int) -> List[Dict[str, Any]]:
        """Use BERT to intelligently extract product information"""
        products = []
        
        # Split text into meaningful chunks
        chunks = self._intelligent_text_chunking(text, query)
        
        for i, chunk in enumerate(chunks[:max_products]):
            try:
                # Use BERT NER to extract entities
                entities = self._extract_entities_with_bert(chunk)
                
                # Check if chunk is relevant to the query
                relevance_score = self._calculate_relevance(chunk, query)
                
                if relevance_score < 0.3:  # Skip irrelevant chunks
                    continue
                
                # Extract product information
                product_info = self._parse_entities_to_product(entities, chunk)
                
                # Assign URLs and images
                if i < len(urls):
                    product_info["url"] = urls[i]
                else:
                    product_info["url"] = base_url
                
                if i < len(images):
                    product_info["image_url"] = images[i]
                
                # Validate and clean product data
                if self._validate_product(product_info):
                    # Determine source from URL
                    source_info = self._identify_source(base_url)
                    product_info.update(source_info)
                    
                    # Generate unique ID
                    product_info["id"] = f"{source_info['source_key']}_{abs(hash(product_info['title'] + str(i)))}"
                    
                    products.append(product_info)
                    logger.info(f"✓ BERT Extracted: {product_info['title']} - {product_info['price']}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        return products
    
    def _intelligent_text_chunking(self, text: str, query: str) -> List[str]:
        """Intelligently split text into product-relevant chunks"""
        # Split by common separators
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        # Keywords that indicate product boundaries
        product_indicators = [
            'price', 'buy', 'add to cart', 'shop', 'product', 'item',
            '₵', '$', '€', '£', 'USD', 'GHS', 'delivery', 'shipping'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            current_chunk.append(line)
            
            # Check if this line indicates a product boundary
            line_lower = line.lower()
            has_price = bool(re.search(r'[\$₵€£]\s*\d+|price.*\d+', line_lower))
            has_product_indicator = any(indicator in line_lower for indicator in product_indicators)
            
            # Create chunk when we have enough content and indicators
            if len(current_chunk) >= 2 and (has_price or has_product_indicator):
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > 50:  # Minimum chunk size
                    chunks.append(chunk_text)
                current_chunk = []
            
            # Prevent chunks from getting too large
            elif len('\n'.join(current_chunk)) > 500:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks[:20]  # Limit number of chunks
    
    def _extract_entities_with_bert(self, text: str) -> List[Dict]:
        """Extract entities using BERT NER model"""
        try:
            if self.ner_pipeline:
                entities = self.ner_pipeline(text)
                return entities
            else:
                # Fallback to rule-based extraction
                return self._fallback_entity_extraction(text)
        except Exception as e:
            logger.error(f"BERT NER failed: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict]:
        """Fallback entity extraction using regex patterns"""
        entities = []
        
        # Extract titles (usually at the beginning or prominent text)
        title_patterns = [
            r'^([A-Za-z0-9\s\-,\.]{10,100})(?:\s*[\$₵€£]|\s*price|\s*buy)',
            r'([A-Za-z0-9\s\-,\.]{15,80})\s*\n'
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                entities.append({
                    'entity_group': 'TITLE',
                    'word': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.8
                })
        
        # Extract prices
        price_patterns = [
            r'([\$₵€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(price\s*:?\s*[\$₵€£]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*[\$₵€£])'
        ]
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'entity_group': 'PRICE',
                    'word': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.9
                })
        
        return entities
    
    def _parse_entities_to_product(self, entities: List[Dict], text: str) -> Dict[str, Any]:
        """Parse BERT entities into product information"""
        product = {
            'title': '',
            'price': 0.0,
            'currency': '',
            'description': text[:200] + '...' if len(text) > 200 else text,
            'url': '',
            'image_url': ''
        }
        
        # Extract title from entities
        title_entities = [e for e in entities if 'TITLE' in e.get('entity_group', '')]
        if title_entities:
            # Use the longest title entity
            best_title = max(title_entities, key=lambda x: len(x['word']))
            product['title'] = self._clean_title(best_title['word'])
        
        # Extract price from entities
        price_entities = [e for e in entities if 'PRICE' in e.get('entity_group', '')]
        if price_entities:
            best_price = max(price_entities, key=lambda x: x.get('score', 0))
            price_info = self._parse_price(best_price['word'])
            product['price'] = price_info['amount']
            product['currency'] = price_info['currency']
        
        # If no entities found, use fallback extraction
        if not product['title']:
            product['title'] = self._extract_title_fallback(text)
        
        if not product['price']:
            price_info = self._extract_price_fallback(text)
            product['price'] = price_info['amount']
            product['currency'] = price_info['currency']
        
        return product
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate how relevant the text is to the search query"""
        try:
            # Use TF-IDF similarity
            corpus = [query.lower(), text.lower()]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Boost score if text contains query words
            query_words = query.lower().split()
            text_lower = text.lower()
            word_matches = sum(1 for word in query_words if word in text_lower)
            word_boost = (word_matches / len(query_words)) * 0.3
            
            return min(similarity + word_boost, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            # Fallback to simple word matching
            query_words = query.lower().split()
            text_lower = text.lower()
            matches = sum(1 for word in query_words if word in text_lower)
            return matches / len(query_words) if query_words else 0.0
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize product title"""
        if not title:
            return ""
        
        # Remove price information from title
        title = re.sub(r'[\$₵€£]\s*\d+.*$', '', title)
        title = re.sub(r'price.*$', '', title, flags=re.IGNORECASE)
        
        # Clean whitespace and special characters
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        
        # Limit length
        return title[:100]
    
    def _parse_price(self, price_str: str) -> Dict[str, Any]:
        """Parse price string to extract amount and currency"""
        if not price_str:
            return {'amount': 0.0, 'currency': ''}
        
        # Currency symbols mapping
        currency_map = {
            '$': 'USD', '₵': 'GHS', '€': 'EUR', '£': 'GBP'
        }
        
        # Extract currency
        currency = ''
        for symbol, code in currency_map.items():
            if symbol in price_str:
                currency = code
                break
        
        # Extract numeric amount
        amount_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', price_str)
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '')
            try:
                amount = float(amount_str)
                return {'amount': amount, 'currency': currency}
            except ValueError:
                pass
        
        return {'amount': 0.0, 'currency': currency}
    
    def _extract_title_fallback(self, text: str) -> str:
        """Fallback title extraction"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Skip lines that look like prices or navigation
                if not re.search(r'[\$₵€£]\d+|price|buy|cart|menu|home', line, re.IGNORECASE):
                    return self._clean_title(line)
        return ""
    
    def _extract_price_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback price extraction"""
        price_patterns = [
            r'([\$₵€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*[\$₵€£])'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                return self._parse_price(match.group(1))
        
        return {'amount': 0.0, 'currency': ''}
    
    def _extract_urls_from_html(self, html: str, base_url: str) -> List[str]:
        """Extract product URLs from HTML"""
        urls = []
        
        # Extract URLs that look like product pages
        url_patterns = [
            r'href=["\']([^"\']*product[^"\']*)["\']',
            r'href=["\']([^"\']*item[^"\']*)["\']',
            r'href=["\']([^"\']*p/[^"\']*)["\']'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                full_url = urljoin(base_url, match)
                if full_url not in urls:
                    urls.append(full_url)
        
        return urls[:20]  # Limit number of URLs
    
    def _extract_images_from_html(self, html: str) -> List[str]:
        """Extract product images from HTML"""
        image_patterns = [
            r'src=["\']([^"\']*\.(?:jpg|jpeg|png|gif|webp)[^"\']*)["\']',
            r'data-src=["\']([^"\']*\.(?:jpg|jpeg|png|gif|webp)[^"\']*)["\']'
        ]
        
        images = []
        for pattern in image_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if 'product' in match.lower() or 'item' in match.lower():
                    images.append(match)
        
        return images[:20]  # Limit number of images
    
    def _validate_product(self, product: Dict[str, Any]) -> bool:
        """Validate that product has minimum required information"""
        return (
            product.get('title', '') and 
            len(product['title']) > 5 and
            product.get('price', 0) > 0
        )
    
    def _identify_source(self, url: str) -> Dict[str, str]:
        """Identify the source website from URL"""
        domain = urlparse(url).netloc.lower()
        
        source_mapping = {
            'jumia.com': {'source': 'Jumia Ghana', 'source_key': 'jumia'},
            'ebay.com': {'source': 'eBay', 'source_key': 'ebay'},
            'aliexpress.com': {'source': 'AliExpress', 'source_key': 'aliexpress'},
            'jiji.com': {'source': 'Jiji Ghana', 'source_key': 'jiji'},
            'tonaton.com': {'source': 'Tonaton', 'source_key': 'tonaton'}
        }
        
        for key, info in source_mapping.items():
            if key in domain:
                return info
        
        return {'source': 'Unknown', 'source_key': 'unknown'}
    
    def _rank_and_filter_products(self, products: List[Dict], query: str) -> List[Dict]:
        """Rank and filter products based on relevance"""
        if not products:
            return products
        
        # Calculate relevance scores
        for product in products:
            title_relevance = self._calculate_relevance(product['title'], query)
            desc_relevance = self._calculate_relevance(product.get('description', ''), query)
            product['relevance_score'] = (title_relevance * 0.7) + (desc_relevance * 0.3)
        
        # Sort by relevance score
        products.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Remove duplicates based on title similarity
        unique_products = []
        seen_titles = set()
        
        for product in products:
            title_lower = product['title'].lower()
            is_duplicate = False
            
            for seen_title in seen_titles:
                if self._calculate_relevance(title_lower, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_products.append(product)
                seen_titles.add(title_lower)
        
        return unique_products

# Global BERT scraper instance
bert_scraper = BERTScraper()

def scrape_with_bert(query: str, max_products: int = 15) -> List[Dict[str, Any]]:
    """
    Main function to scrape products using BERT intelligence
    
    Args:
        query: Search query
        max_products: Maximum number of products to return
    
    Returns:
        List of product dictionaries
    """
    return bert_scraper.scrape_products(query, max_products=max_products)

"""
BERT Product Scraper
Main module for extracting product information using the trained BERT model
"""

import os
import re
import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from playwright.async_api import async_playwright
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTProductScraper:
    """BERT-powered product information extractor"""
    
    def __init__(self, model_dir: str = "finetuned-bert-product"):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        # Entity labels mapping
        self.label_mapping = {
            'B-TITLE': 'title_start',
            'I-TITLE': 'title_continue',
            'B-PRICE': 'price_start', 
            'I-PRICE': 'price_continue',
            'B-URL': 'url_start',
            'I-URL': 'url_continue',
            'B-IMG': 'img_start',
            'I-IMG': 'img_continue',
            'B-RATING': 'rating_start',
            'I-RATING': 'rating_continue',
            'B-AVAILABILITY': 'availability_start',
            'I-AVAILABILITY': 'availability_continue',
            'O': 'other'
        }
        
    def load_model(self) -> bool:
        """Load the trained BERT model and tokenizer"""
        try:
            if not os.path.exists(self.model_dir):
                logger.error(f"Model directory not found: {self.model_dir}")
                return False
                
            logger.info("Loading BERT model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
            self.is_loaded = True
            logger.info("BERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            return False
    
    def extract_products_from_html(self, html_content: str, max_length: int = 512) -> List[Dict[str, Any]]:
        """Extract product information from HTML using BERT model"""
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        try:
            # Clean HTML content for better processing
            cleaned_html = self._clean_html(html_content)
            
            # Split content into chunks if too long
            chunks = self._split_content(cleaned_html, max_length)
            
            all_products = []
            for chunk in chunks:
                products = self._extract_from_chunk(chunk)
                all_products.extend(products)
            
            # Remove duplicates and merge similar products
            unique_products = self._deduplicate_products(all_products)
            
            logger.info(f"Extracted {len(unique_products)} unique products from HTML")
            return unique_products
            
        except Exception as e:
            logger.error(f"Error extracting products from HTML: {e}")
            return []
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content for better BERT processing"""
        # Remove script and style tags
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove excessive whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content.strip()
    
    def _split_content(self, content: str, max_length: int) -> List[str]:
        """Split content into chunks that fit BERT's token limit"""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_length * 3
        
        if len(content) <= max_chars:
            return [content]
        
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + max_chars, len(content))
            # Try to break at word boundary
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(content[start:end])
            start = end
        
        return chunks
    
    def _extract_from_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        """Extract products from a single chunk using BERT"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                chunk, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Decode predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.model.config.id2label[pred.item()] for pred in predictions[0]]
            
            # Extract entities
            products = self._extract_entities(tokens, predicted_labels, chunk)
            return products
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return []
    
    def _extract_entities(self, tokens: List[str], labels: List[str], original_text: str) -> List[Dict[str, Any]]:
        """Extract product entities from tokens and labels"""
        products = []
        current_product = {}
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label.startswith('B-'):  # Beginning of entity
                # Save previous entity
                if current_entity and current_tokens:
                    entity_text = self._reconstruct_text(current_tokens)
                    current_product[current_entity] = entity_text
                
                # Start new entity
                current_entity = label[2:].lower()  # Remove 'B-' prefix
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity:  # Inside entity
                current_tokens.append(token)
                
            elif label == 'O':  # Outside entity
                # Save current entity if exists
                if current_entity and current_tokens:
                    entity_text = self._reconstruct_text(current_tokens)
                    current_product[current_entity] = entity_text
                    current_entity = None
                    current_tokens = []
                
                # If we have a complete product, save it
                if current_product and ('title' in current_product or 'price' in current_product):
                    products.append(current_product.copy())
                    current_product = {}
        
        # Save final entity and product
        if current_entity and current_tokens:
            entity_text = self._reconstruct_text(current_tokens)
            current_product[current_entity] = entity_text
            
        if current_product and ('title' in current_product or 'price' in current_product):
            products.append(current_product)
        
        return products
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from BERT tokens"""
        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]  # Remove ## prefix for subwords
            else:
                if text:
                    text += " "
                text += token
        return text.strip()
    
    def _deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate products based on title and price similarity"""
        if not products:
            return []
        
        unique_products = []
        seen_products = set()
        
        for product in products:
            # Create a signature for the product
            title = product.get('title', '').strip().lower()
            price = product.get('price', '').strip().lower()
            
            # Skip products without title or price
            if not title and not price:
                continue
            
            signature = f"{title[:50]}|{price}"
            
            if signature not in seen_products:
                seen_products.add(signature)
                unique_products.append(product)
        
        return unique_products

    async def scrape_with_playwright(self, url: str, site_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape a URL using Playwright and extract products with BERT"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                page = await context.new_page()
                
                # Navigate to URL
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for content to load
                if 'wait_for' in site_config:
                    try:
                        await page.wait_for_selector(site_config['wait_for'], timeout=10000)
                    except:
                        logger.warning(f"Wait selector not found: {site_config['wait_for']}")
                
                # Get page content
                html_content = await page.content()
                
                await browser.close()
                
                # Extract products using BERT
                products = self.extract_products_from_html(html_content)
                
                # Add source information
                for product in products:
                    product['source'] = site_config.get('name', 'unknown')
                    product['url'] = url
                
                return products
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []

# Global instance
bert_product_scraper = BERTProductScraper()
