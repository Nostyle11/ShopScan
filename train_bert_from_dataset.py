"""
Train BERT Model from Existing Dataset
Loads bert_training_dataset.json and trains the BERT model directly
"""

import json
import logging
import os
from enhanced_bert_trainer import EnhancedBERTTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_bert_from_existing_dataset(dataset_file: str = "enhanced_bert_training_dataset.json"):
    """Load existing dataset and train BERT model"""
    
    # Check if enhanced dataset exists, fallback to original
    if not os.path.exists(dataset_file):
        fallback_file = "bert_training_dataset.json"
        if os.path.exists(fallback_file):
            print(f"⚠️  Enhanced dataset '{dataset_file}' not found, using fallback: {fallback_file}")
            dataset_file = fallback_file
        else:
            print(f"❌ No dataset files found! Tried: {dataset_file}, {fallback_file}")
            return False
    
    try:
        # Load the existing dataset
        print(f"📂 Loading training dataset from {dataset_file}...")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Handle different dataset formats
        if isinstance(dataset, dict) and 'training_data' in dataset:
            # Enhanced dataset format with metadata
            training_data = dataset['training_data']
            metadata = dataset.get('metadata', {})
            print(f"📊 Enhanced dataset loaded:")
            print(f"   Total examples: {metadata.get('total_examples', len(training_data))}")
            print(f"   Sites: {metadata.get('sites', 'Unknown')}")
            print(f"   Collection date: {metadata.get('collection_date', 'Unknown')}")
        elif isinstance(dataset, list):
            # Original dataset format (list of examples)
            training_data = dataset
            print(f"📊 Original dataset loaded: {len(training_data)} examples")
        else:
            print(f"❌ Unknown dataset format in {dataset_file}")
            return False
        
        print(f"✅ Loaded {len(training_data)} training examples")
        
        # Initialize BERT trainer
        print("🧠 Initializing BERT trainer...")
        trainer = EnhancedBERTTrainer()
        
        # Clear existing training data and load our dataset
        trainer.training_data = []
        
        # Convert dataset format if needed
        for entry in training_data:
            # Check if entry has the required format
            if 'text' in entry and 'query' in entry:
                trainer_entry = {
                    'text': entry['text'],  # Already has [TITLE]...[/TITLE] format
                    'query': entry['query'],
                    'source': entry.get('site', 'unknown'),
                    'timestamp': entry.get('timestamp', ''),
                    'product_data': entry.get('product_data', {})
                }
                trainer.training_data.append(trainer_entry)
            else:
                logger.warning(f"Skipping entry with missing required fields: {entry.keys()}")
        
        print(f"📊 Prepared {len(trainer.training_data)} examples for training")
        
        # Save the training data to the enhanced trainer's file
        trainer.save_training_data()
        
        # Check if we have enough data for training
        if len(trainer.training_data) < 10:
            print(f"❌ Not enough training data: {len(trainer.training_data)} examples (minimum 10 required)")
            return False
        
        # Show statistics
        stats = trainer.get_training_stats()
        print(f"📈 Training Statistics:")
        print(f"   Total examples: {stats['total_examples']}")
        print(f"   Sources: {list(stats['sources'].keys())}")
        print(f"   Queries: {len(stats['queries'])} unique queries")
        
        # Train the model with conservative settings
        print("🚀 Starting BERT model training...")
        print("   Using conservative settings: epochs=2, batch_size=8")
        
        success = trainer.train_model(epochs=2, batch_size=8, learning_rate=2e-5)
        
        if success:
            print("🎉 BERT model training completed successfully!")
            print(f"💾 Model saved to: {trainer.model_dir}")
            print("✅ Ready for production use!")
            return True
        else:
            print("❌ BERT model training failed!")
            return False
            
    except FileNotFoundError:
        print(f"❌ Dataset file '{dataset_file}' not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON dataset: {e}")
        return False
    except ImportError as e:
        print(f"❌ Could not import EnhancedBERTTrainer: {e}")
        print("💡 Make sure enhanced_bert_trainer.py is available")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔥 BERT Model Training from Existing Dataset")
    print("=" * 50)
    
    success = train_bert_from_existing_dataset()
    
    if success:
        print("\n🎊 Training Complete!")
        print("Your BERT model is now ready to use for intelligent product extraction.")
    else:
        print("\n💔 Training Failed!")
        print("Check the error messages above for troubleshooting.")
