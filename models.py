from datetime import datetime
from app import db

class Source(db.Model):
    """Represents an e-commerce source/website"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    logo = db.Column(db.String(100), nullable=True)
    active = db.Column(db.Boolean, default=True)
    price_entries = db.relationship('PriceEntry', backref='source', lazy=True)
    
    def __repr__(self):
        return f'<Source {self.name}>'


class Product(db.Model):
    """Represents a product that can be sold on multiple websites"""
    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    image_url = db.Column(db.String(1024), nullable=True)
    description = db.Column(db.Text, nullable=True)
    price_entries = db.relationship('PriceEntry', backref='product', lazy=True)
    date_added = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<Product {self.name}>'
    
    def get_best_price(self):
        """Returns the price entry with the lowest price"""
        if not self.price_entries:
            return None
        return min(self.price_entries, key=lambda x: x.price)


class PriceEntry(db.Model):
    """Represents a price for a product from a specific source"""
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(50), db.ForeignKey('product.id'), nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey('source.id'), nullable=False)
    price = db.Column(db.Float, nullable=False)
    url = db.Column(db.String(1024), nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<PriceEntry {self.product_id} from {self.source_id}: ${self.price}>'
