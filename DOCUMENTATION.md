# Price Comparison Web Application - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [Scraping System](#scraping-system)
6. [API Reference](#api-reference)
7. [Database Schema](#database-schema)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Development Guidelines](#development-guidelines)

## Project Overview

A Flask-based price comparison web application that allows users to:
- Upload product images for AI-powered product identification
- Search for products across multiple e-commerce platforms
- Compare prices from different retailers
- View product details with images and links
- Cache search results for improved performance

### Key Features
- **AI Image Analysis**: OpenAI Vision API identifies products from uploaded images
- **Multi-Site Scraping**: Real-time price comparison across 5 e-commerce platforms
- **Modular Architecture**: Each site has dedicated scraper module
- **BERT Intelligence**: AI-powered content extraction and product matching
- **Progressive Web App**: Installable app with offline capabilities
- **Responsive Design**: Mobile-first interface with Bootstrap
- **Currency Conversion**: Automatic conversion to Ghana Cedis (GHS)

### Supported E-commerce Sites
1. **eBay** - Global marketplace with extensive product catalog
2. **Jiji Ghana** - Local Ghanaian marketplace
3. **Jumia Ghana** - African e-commerce platform
4. **Tonaton** - Ghanaian classified ads platform
5. **AliExpress** - Global marketplace for affordable products

## Architecture

### High-Level Architecture
```
User Interface (Bootstrap + HTML)
    ↓
Flask Application (routes.py)
    ↓
ScraperManager (coordinates all scrapers)
    ↓
Individual Site Scrapers (modular)
    ↓
HTTP → URL Extraction → Trafilatura → BERT → JSON
    ↓
PostgreSQL Database Storage
```

### Core Workflow
1. **User Input**: Search query or image upload
2. **Image Analysis** (if image): OpenAI Vision API identifies product
3. **Query Processing**: Search term is prepared for each site
4. **Parallel Scraping**: All site scrapers run simultaneously
5. **Data Processing**: Extract URLs, clean text, analyze with BERT
6. **Result Aggregation**: Combine and format results
7. **Database Storage**: Save products and prices
8. **Display**: Show results with currency conversion

### Directory Structure
```
/
├── app.py                    # Flask application initialization
├── main.py                   # Application entry point
├── routes.py                 # Web routes and API endpoints
├── models.py                 # Database models
├── cache.py                  # Result caching system
├── currency_converter.py     # Price conversion utilities
├── vision_analyzer.py        # OpenAI Vision API integration
├── scrapers/                 # Modular scraper system
│   ├── scraper_manager.py    # Central coordinator
│   ├── ebay_scraper.py       # eBay-specific scraper
│   ├── jiji_scraper.py       # Jiji Ghana scraper
│   ├── jumia_scraper.py      # Jumia Ghana scraper
│   ├── tonaton_scraper.py    # Tonaton scraper
│   └── aliexpress_scraper.py # AliExpress scraper
├── static/                   # Static assets
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript files
│   ├── icons/                # App icons
│   └── manifest.json         # PWA manifest
├── templates/                # HTML templates
│   ├── layout.html           # Base template
│   ├── index.html            # Homepage
│   └── search_results.html   # Results page
└── training_data.json        # BERT training data
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL database
- OpenAI API key

### Environment Variables
```bash
DATABASE_URL=postgresql://username:password@host:port/database
OPENAI_API_KEY=your_openai_api_key_here
SESSION_SECRET=your_session_secret_here
```

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Initialize database: `python -c "from app import db; db.create_all()"`
5. Start application: `gunicorn --bind 0.0.0.0:5000 main:app`

### Dependencies
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **OpenAI**: AI image analysis
- **Requests**: HTTP client for scraping
- **Trafilatura**: HTML text extraction
- **BeautifulSoup4**: HTML parsing
- **Scikit-learn**: BERT-style text analysis
- **Numpy**: Numerical operations

## Core Components

### 1. Flask Application (`app.py`)
Main application factory with database initialization:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
db = SQLAlchemy(app)
```

### 2. Routes (`routes.py`)
Web endpoints and API handlers:
- `GET /` - Homepage with search form
- `POST /search` - Product search with filters
- `POST /api/analyze-image` - Image analysis endpoint
- `GET /product/<id>` - Product details page
- `POST /refresh/<id>` - Refresh product prices

### 3. ScraperManager (`scrapers/scraper_manager.py`)
Central coordinator for all site scrapers:
```python
class ScraperManager:
    def __init__(self):
        self.scrapers = {
            'ebay': EbayScraper(),
            'jiji': JijiScraper(),
            'jumia': JumiaScraper(),
            'tonaton': TonatonScraper(),
            'aliexpress': AliexpressScraper()
        }
    
    def scrape_all_sites(self, query, products_per_site=4):
        # Coordinate parallel scraping across all sites
```

### 4. Individual Scrapers
Each scraper follows identical 5-step workflow:
1. **HTTP Request**: Direct request to search URL
2. **URL Extraction**: Parse URLs and images from raw HTML
3. **Text Cleaning**: Use Trafilatura to extract clean content
4. **BERT Analysis**: Extract structured product information
5. **JSON Output**: Format results consistently

### 5. Database Models (`models.py`)
Three main entities:
- **Product**: Core product information
- **Source**: E-commerce platform details
- **PriceEntry**: Price records with timestamps

## Scraping System

### 5-Step Workflow
Each site scraper implements this exact workflow:

#### Step 1-2: HTTP Request
```python
response = requests.get(search_url, headers=headers, timeout=10)
```

#### Step 3a: URL/Image Extraction
```python
soup = BeautifulSoup(response.content, 'html.parser')
urls = [urljoin(base_url, link.get('href')) for link in soup.find_all('a', href=True)]
images = [img.get('src') for img in soup.find_all('img', src=True)]
```

#### Step 3b: Text Cleaning
```python
import trafilatura
clean_text = trafilatura.extract(response.text)
```

#### Step 4: BERT Analysis
```python
from simple_bert_training import SimpleBERTTrainer
trainer = SimpleBERTTrainer()
result = trainer.predict(clean_text)
```

#### Step 5: JSON Formatting
```python
return {
    'id': unique_id,
    'title': extracted_title,
    'price': extracted_price,
    'image_url': assigned_image,
    'url': assigned_url,
    'source': site_name
}
```

### Site-Specific Details

#### eBay Scraper
- **URL**: `https://www.ebay.com/sch/i.html?_nkw={query}`
- **Features**: Large product catalog, reliable structure
- **Currency**: USD (converted to GHS)
- **Expected Products**: 10 per search

#### Jiji Ghana Scraper
- **URL**: `https://jiji.com.gh/search?query={query}`
- **Features**: Local Ghanaian products
- **Currency**: GHS
- **Status**: May encounter 403 errors due to bot protection

#### Jumia Ghana Scraper
- **URL**: `https://www.jumia.com.gh/catalog/?q={query}`
- **Features**: African e-commerce platform
- **Currency**: GHS
- **Structure**: Product cards with consistent layout

#### Tonaton Scraper
- **URL**: `https://tonaton.com/s_{query}-in-ghana`
- **Features**: Classified ads format
- **Currency**: GHS
- **Status**: May encounter 403 errors

#### AliExpress Scraper
- **URL**: `https://www.aliexpress.com/wholesale?SearchText={query}`
- **Features**: Global marketplace, affordable prices
- **Currency**: USD (converted to GHS)
- **Challenges**: Heavy anti-bot protection

## API Reference

### Image Analysis API
**Endpoint**: `POST /api/analyze-image`

**Request**:
```json
{
    "image": "base64_encoded_image_data"
}
```

**Response**:
```json
{
    "success": true,
    "query": "iPhone 13 Pro Max",
    "confidence": 0.95,
    "brand": "Apple",
    "category": "Electronics",
    "features": ["128GB", "Blue", "Unlocked"]
}
```

### Search API
**Endpoint**: `POST /search`

**Parameters**:
- `query`: Search term
- `min_price`: Minimum price filter
- `max_price`: Maximum price filter
- `category`: Product category
- `selected_retailers`: Array of retailer keys

**Response**: HTML page with search results

## Database Schema

### Products Table
```sql
CREATE TABLE product (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    image_url VARCHAR,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Sources Table
```sql
CREATE TABLE source (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    url VARCHAR,
    logo VARCHAR,
    active BOOLEAN DEFAULT TRUE
);
```

### Price Entries Table
```sql
CREATE TABLE price_entry (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR REFERENCES product(id),
    source_id INTEGER REFERENCES source(id),
    price DECIMAL(10,2) NOT NULL,
    url VARCHAR,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Cache Settings
- **Default TTL**: 1 hour
- **Storage**: In-memory dictionary
- **Key Format**: `{query}_{category}_{min_price}_{max_price}_{retailers}`

### Rate Limiting
- **Request Timeout**: 10 seconds per site
- **Retry Logic**: None (fail fast)
- **Concurrent Requests**: All sites scraped in parallel

### Currency Conversion
- **Base Currency**: Ghana Cedis (GHS)
- **Exchange Rate**: 1 USD = 12 GHS (configurable)
- **Format**: ₵{amount}

## Troubleshooting

### Common Issues

#### 403 Forbidden Errors
**Cause**: Anti-bot protection on target sites
**Solution**: Sites implement bot detection; this is expected behavior

#### Empty Results
**Cause**: BERT analysis doesn't find valid products
**Solution**: Adjust BERT training data or search terms

#### Slow Performance
**Cause**: Network timeouts or heavy pages
**Solution**: Sites timeout after 10 seconds automatically

#### Database Connection Issues
**Cause**: Missing DATABASE_URL environment variable
**Solution**: Set PostgreSQL connection string

### Debugging Tools

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Individual Scrapers
```python
from scrapers.ebay_scraper import EbayScraper
scraper = EbayScraper()
results = scraper.scrape('iPhone 13', 5)
```

#### Check Database Connection
```python
from app import db
db.engine.execute('SELECT 1')
```

## Development Guidelines

### Adding New E-commerce Sites

1. **Create Scraper File**: `scrapers/newsite_scraper.py`
2. **Implement Base Class**: Inherit from common interface
3. **Follow 5-Step Workflow**: HTTP → URLs → Trafilatura → BERT → JSON
4. **Register in Manager**: Add to ScraperManager.scrapers dict
5. **Test Thoroughly**: Verify real data extraction

### Code Style
- **Python**: PEP 8 compliance
- **Naming**: Snake_case for functions, PascalCase for classes
- **Comments**: Document complex logic and site-specific quirks
- **Error Handling**: Fail gracefully with informative logs

### Testing Strategy
- **Unit Tests**: Individual scraper functions
- **Integration Tests**: Full search workflow
- **Manual Testing**: Real searches on live sites
- **Performance Tests**: Timeout and concurrency limits

### Security Considerations
- **Input Validation**: Sanitize all user inputs
- **Rate Limiting**: Respect target site policies
- **Error Messages**: Don't expose internal details
- **Environment Variables**: Keep secrets secure

## Performance Optimization

### Caching Strategy
- **Search Results**: Cache for 1 hour
- **Image Analysis**: Cache product identification
- **Database Queries**: Use SQLAlchemy query optimization

### Scalability Considerations
- **Concurrent Scraping**: All sites scraped simultaneously
- **Database Indexing**: Index on product names and prices
- **Static Assets**: Serve via CDN in production
- **Session Management**: Stateless design for horizontal scaling

## Deployment

### Production Configuration
```python
# Gunicorn settings
workers = 2
worker_class = "sync"
bind = "0.0.0.0:5000"
timeout = 30
```

### Environment Setup
- **Database**: PostgreSQL with connection pooling
- **Reverse Proxy**: Nginx for static files
- **SSL**: HTTPS termination at load balancer
- **Monitoring**: Application logs and metrics

### Health Checks
- **Database**: Connection test on startup
- **Scrapers**: Periodic validation of target sites
- **Memory**: Monitor for cache size growth
- **Response Times**: Track search performance

This documentation provides comprehensive coverage of the price comparison application architecture, implementation details, and operational procedures.