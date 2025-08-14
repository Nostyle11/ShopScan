"""
Enhanced BERT Training System
Continuously learns from web scraping data to improve product extraction
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ProductDataset(Dataset):
    """Custom dataset for product entity recognition"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=True if isinstance(text, list) else False
        )
        
        # Align labels with tokenized input
        if isinstance(text, list):  # Pre-tokenized
            word_ids = encoding.word_ids()
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    aligned_labels.append(labels[word_idx] if word_idx < len(labels) else 0)
                else:
                    aligned_labels.append(-100)  # Subword token
                previous_word_idx = word_idx
        else:
            # For non-tokenized text, create simple alignment
            aligned_labels = labels[:self.max_length] + [-100] * (self.max_length - len(labels))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class EnhancedBERTTrainer:
    """Enhanced BERT trainer for continuous learning from web data"""
    
    def __init__(self, model_dir: str = "finetuned-bert-product"):
        self.model_dir = model_dir
        self.training_data_file = os.path.join(model_dir, "enhanced_training_data.json")
        self.model_config_file = os.path.join(model_dir, "model_config.json")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Label mapping for entity recognition
        self.label2id = {
            "O": 0,           # Outside entity
            "B-TITLE": 1,     # Beginning of title
            "I-TITLE": 2,     # Inside title
            "B-PRICE": 3,     # Beginning of price
            "I-PRICE": 4,     # Inside price
            "B-URL": 5,       # Beginning of URL
            "I-URL": 6,       # Inside URL
            "B-IMG": 7,       # Beginning of image URL
            "I-IMG": 8,       # Inside image URL
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Training data storage
        self.training_data = []
        self.load_training_data()
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        logger.info("Enhanced BERT Trainer initialized")
    
    def add_web_scraping_data(self, scraped_products: List[Dict[str, Any]], query: str):
        """Add data from web scraping results for training"""
        for product in scraped_products:
            try:
                # Create training example from product data
                training_example = self._create_training_example_from_product(product, query)
                if training_example:
                    self.training_data.append(training_example)
                    logger.info(f"Added training data for: {product.get('title', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error adding training data: {e}")
        
        # Save updated training data
        self.save_training_data()
    
    def _create_training_example_from_product(self, product: Dict[str, Any], query: str) -> Optional[Dict]:
        """Create a training example from a scraped product"""
        title = product.get('title', '')
        price = product.get('price', 0)
        url = product.get('url', '')
        image_url = product.get('image_url', '')
        description = product.get('description', '')
        
        if not title or not price:
            return None
        
        # Create annotated text with entity tags
        annotated_text = f"Query: {query}\n"
        annotated_text += f"[TITLE]{title}[/TITLE] "
        
        if price:
            price_str = f"{product.get('currency', '')} {price}".strip()
            annotated_text += f"[PRICE]{price_str}[/PRICE] "
        
        if url:
            annotated_text += f"[URL]{url}[/URL] "
        
        if image_url:
            annotated_text += f"[IMG]{image_url}[/IMG] "
        
        # Add description context
        if description:
            annotated_text += f"Description: {description[:100]}"
        
        return {
            'text': annotated_text,
            'query': query,
            'product_data': product,
            'timestamp': datetime.now().isoformat(),
            'source': product.get('source', 'unknown')
        }
    
    def prepare_training_data(self) -> Tuple[List[str], List[List[int]]]:
        """Prepare training data for BERT model"""
        texts = []
        labels = []
        
        for example in self.training_data:
            try:
                text = example['text']
                tokens, token_labels = self._extract_tokens_and_labels(text)
                
                if tokens and token_labels and len(tokens) == len(token_labels):
                    texts.append(tokens)
                    labels.append(token_labels)
            except Exception as e:
                logger.error(f"Error preparing training example: {e}")
                continue
        
        return texts, labels
    
    def _extract_tokens_and_labels(self, text: str) -> Tuple[List[str], List[int]]:
        """Extract tokens and their corresponding labels from annotated text"""
        tokens = []
        labels = []
        
        # Find all entity spans
        entity_pattern = r'\[(\w+)\](.*?)\[/\1\]'
        spans = []
        
        for match in re.finditer(entity_pattern, text):
            entity_type = match.group(1)
            content = match.group(2).strip()
            start = match.start()
            end = match.end()
            spans.append((start, end, entity_type, content))
        
        # Sort spans by position
        spans.sort(key=lambda x: x[0])
        
        # Process text with entity annotations
        current_pos = 0
        
        for start, end, entity_type, content in spans:
            # Add text before entity as "O" (outside)
            if current_pos < start:
                prefix_text = text[current_pos:start].strip()
                if prefix_text:
                    prefix_tokens = prefix_text.split()
                    tokens.extend(prefix_tokens)
                    labels.extend([self.label2id["O"]] * len(prefix_tokens))
            
            # Add entity tokens with appropriate labels
            entity_tokens = content.split()
            if entity_tokens:
                tokens.extend(entity_tokens)
                # First token gets B- label, rest get I- labels
                b_label = f"B-{entity_type}"
                i_label = f"I-{entity_type}"
                
                if b_label in self.label2id:
                    labels.append(self.label2id[b_label])
                    if len(entity_tokens) > 1:
                        labels.extend([self.label2id.get(i_label, self.label2id["O"])] * (len(entity_tokens) - 1))
                else:
                    # Fallback to "O" if entity type not recognized
                    labels.extend([self.label2id["O"]] * len(entity_tokens))
            
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                remaining_tokens = remaining_text.split()
                tokens.extend(remaining_tokens)
                labels.extend([self.label2id["O"]] * len(remaining_tokens))
        
        return tokens, labels
    
    def train_model(self, epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the BERT model with current training data"""
        if len(self.training_data) < 10:
            logger.warning(f"Not enough training data: {len(self.training_data)} examples")
            return False
        
        try:
            # Prepare training data
            texts, labels = self.prepare_training_data()
            
            if not texts:
                logger.error("No valid training data prepared")
                return False
            
            logger.info(f"Training with {len(texts)} examples")
            
            # Initialize tokenizer and model
            model_name = "bert-base-cased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = ProductDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = ProductDataset(val_texts, val_labels, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.model_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=os.path.join(self.model_dir, 'logs'),
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None  # Disable wandb/tensorboard
            )
            
            # Data collator
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            # Train the model
            logger.info("Starting BERT training...")
            trainer.train()
            
            # Save the trained model
            logger.info("Saving trained model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.model_dir)
            
            # Save model configuration
            self._save_model_config(epochs, batch_size, learning_rate, len(texts))
            
            logger.info("BERT training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _save_model_config(self, epochs: int, batch_size: int, learning_rate: float, num_examples: int):
        """Save model training configuration"""
        config = {
            'model_type': 'bert-base-cased',
            'num_labels': len(self.label2id),
            'label2id': self.label2id,
            'id2label': self.id2label,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_examples': num_examples
            },
            'last_trained': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(self.model_config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_training_data(self):
        """Save training data to disk"""
        try:
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.training_data)} training examples")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def load_training_data(self):
        """Load training data from disk"""
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                logger.info(f"Loaded {len(self.training_data)} training examples")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                self.training_data = []
        else:
            self.training_data = []
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about training data"""
        if not self.training_data:
            return {'total_examples': 0}
        
        stats = {
            'total_examples': len(self.training_data),
            'sources': {},
            'queries': {},
            'date_range': {
                'earliest': None,
                'latest': None
            }
        }
        
        dates = []
        for example in self.training_data:
            # Count sources
            source = example.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Count queries
            query = example.get('query', 'unknown')
            stats['queries'][query] = stats['queries'].get(query, 0) + 1
            
            # Collect dates
            if 'timestamp' in example:
                dates.append(example['timestamp'])
        
        if dates:
            dates.sort()
            stats['date_range']['earliest'] = dates[0]
            stats['date_range']['latest'] = dates[-1]
        
        return stats
    
    def auto_train_from_scraping(self, query: str, max_products: int = 20):
        """Automatically scrape data and train the model"""
        try:
            # Import the BERT scraper
            from bert_scraper import scrape_with_bert
            
            logger.info(f"Auto-training: Scraping data for query '{query}'")
            
            # Scrape products
            products = scrape_with_bert(query, max_products)
            
            if not products:
                logger.warning("No products scraped for training")
                return False
            
            # Add scraped data to training set
            self.add_web_scraping_data(products, query)
            
            # Train model if we have enough data
            if len(self.training_data) >= 20:
                logger.info("Starting automatic training...")
                return self.train_model()
            else:
                logger.info(f"Need more data for training: {len(self.training_data)}/20")
                return False
                
        except Exception as e:
            logger.error(f"Error in auto-training: {e}")
            return False

# Global enhanced trainer instance
enhanced_trainer = EnhancedBERTTrainer()

def train_bert_from_scraping(query: str, max_products: int = 20) -> bool:
    """
    Train BERT model using data scraped from the web
    
    Args:
        query: Search query to scrape data for
        max_products: Maximum products to scrape
    
    Returns:
        True if training was successful
    """
    return enhanced_trainer.auto_train_from_scraping(query, max_products)

def get_training_statistics() -> Dict[str, Any]:
    """Get current training data statistics"""
    return enhanced_trainer.get_training_stats()

def manual_train_bert(epochs: int = 3, batch_size: int = 16) -> bool:
    """Manually train BERT with current training data"""
    return enhanced_trainer.train_model(epochs, batch_size)
