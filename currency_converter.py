import re
import logging

logger = logging.getLogger(__name__)

def format_price_with_currency(price, currency='GHS'):
    """Format price with currency symbol"""
    try:
        if isinstance(price, str):
            # Extract numeric part
            numeric_price = get_numeric_price(price)
            if numeric_price > 0:
                return f"₵ {numeric_price:,.2f}"
            return price
        elif isinstance(price, (int, float)):
            if currency == 'GHS':
                return f"₵ {price:,.2f}"
            else:
                return f"${price:,.2f}"
        return str(price)
    except Exception as e:
        logger.error(f"Error formatting price {price}: {e}")
        return str(price)

def get_numeric_price(price_text):
    """Extract numeric price from text"""
    try:
        if not price_text:
            return 0.0
        
        # Remove currency symbols and extract numbers
        price_clean = re.sub(r'[₵$£€,]', '', str(price_text))
        price_match = re.search(r'[\d.]+', price_clean)
        
        if price_match:
            return float(price_match.group())
        return 0.0
    except Exception as e:
        logger.error(f"Error extracting numeric price from {price_text}: {e}")
        return 0.0

def convert_usd_to_ghs(usd_price, exchange_rate=10.85):
    """Convert USD to GHS"""
    try:
        numeric_price = get_numeric_price(usd_price) if isinstance(usd_price, str) else usd_price
        return numeric_price * exchange_rate
    except Exception as e:
        logger.error(f"Error converting USD to GHS: {e}")
        return 0.0