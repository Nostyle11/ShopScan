import logging
import base64
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app, db
from models import Product, PriceEntry, Source
from scrapers.scraper_manager import ScraperManager
from vision_analyzer import analyze_product_image, generate_search_query
from cache import get_cached_results, cache_results, invalidate_cache
from currency_converter import format_price_with_currency, get_numeric_price

logger = logging.getLogger(__name__)

def apply_filters(products, category, min_price, max_price, selected_retailers):
    """Apply filters to product list"""
    filtered_products = []
    
    for product in products:
        # Price filtering
        if min_price is not None and product['price'] < min_price:
            continue
        if max_price is not None and product['price'] > max_price:
            continue
            
        # Retailer filtering
        if selected_retailers and product['source_key'] not in selected_retailers:
            continue
            
        # Category filtering (basic keyword matching)
        if category:
            product_title_lower = product['title'].lower()
            category_keywords = {
                'electronics': ['phone', 'laptop', 'computer', 'tablet', 'camera', 'tv', 'electronics', 'iphone', 'samsung', 'apple'],
                'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'clothing', 'fashion', 'wear'],
                'home': ['furniture', 'kitchen', 'home', 'garden', 'decor', 'appliance'],
                'sports': ['sport', 'fitness', 'exercise', 'outdoor', 'bike', 'ball'],
                'books': ['book', 'novel', 'magazine', 'ebook', 'literature'],
                'toys': ['toy', 'game', 'puzzle', 'doll', 'action figure'],
                'automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'motorcycle'],
                'health': ['health', 'beauty', 'cosmetic', 'skincare', 'supplement'],
                'jewelry': ['jewelry', 'watch', 'necklace', 'ring', 'bracelet'],
                'office': ['office', 'business', 'supplies', 'desk', 'chair', 'pen']
            }
            
            if category in category_keywords:
                keywords = category_keywords[category]
                if not any(keyword in product_title_lower for keyword in keywords):
                    continue
        
        filtered_products.append(product)
    
    return filtered_products

def scrape_filtered_sites(query, selected_retailers):
    """Scrape only selected e-commerce sites for products using modular scrapers"""
    # Use modular scraper manager directly
    manager = ScraperManager()
    products_per_site = max(1, 20 // len(manager.get_available_sites()))
    return manager.scrape_all_sites(query, products_per_site)[:20]

@app.route('/')
def index():
    """Homepage with search form"""
    return render_template('index.html')


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image_api():
    """AJAX endpoint to analyze image and return product details"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Please upload JPG, JPEG, PNG, or GIF images.'
            }), 400
        
        # Reset file pointer to beginning
        file.seek(0)
        
        # Analyze image using the new robust vision analyzer
        logger.info(f"Analyzing uploaded image: {file.filename}")
        analysis_result = analyze_product_image(file)
        
        # Check if analysis was successful
        if not analysis_result.get('success'):
            return jsonify({
                'success': False,
                'error': analysis_result.get('error', 'Image analysis failed')
            }), 400
        
        # Generate search query from analysis
        search_query = generate_search_query(analysis_result)
        logger.info(f"Generated search query: {search_query}")
        
        # Return successful analysis result
        return jsonify({
            'success': True,
            'analysis': {
                'product_name': analysis_result.get('product_name', ''),
                'brand': analysis_result.get('brand'),
                'category': analysis_result.get('category'),
                'features': analysis_result.get('features', []),
                'confidence': analysis_result.get('confidence', 0.0)
            },
            'search_query': search_query
        })
            
    except Exception as e:
        logger.error(f"Unexpected error in image analysis API: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500


def allowed_file(filename):
    """Check if uploaded file is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for products across multiple e-commerce sites"""
    from flask import session
    image_analysis = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        category = request.form.get('category', '').strip()
        min_price = request.form.get('min_price', type=float)
        max_price = request.form.get('max_price', type=float)
        selected_retailers = request.form.getlist('retailers')
    else:
        query = request.args.get('query', '').strip()
        category = request.args.get('category', '').strip()
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        selected_retailers = request.args.getlist('retailers')
        is_image_search = request.args.get('image_search', False)
        
        # Handle image analysis data passed via URL parameters
        if is_image_search:
            image_analysis = {
                'product_name': request.args.get('detected_product', ''),
                'brand': request.args.get('detected_brand'),
                'category': request.args.get('detected_category'),
                'features': request.args.get('detected_features', '').split(',') if request.args.get('detected_features') else [],
                'confidence': float(request.args.get('confidence', 0.0))
            }
    
    if not query:
        flash('Please enter a search term', 'warning')
        return redirect(url_for('index'))
    
    # Create cache key including filters
    retailer_key = '-'.join(sorted(selected_retailers)) if selected_retailers else 'all'
    cache_key = f"{query}_{category or 'all'}_{min_price or 0}_{max_price or 99999}_{retailer_key}"
    
    # Check cache first
    cached_result = get_cached_results(cache_key)
    if cached_result:
        products, last_updated = cached_result
        logger.info(f"Returning cached results for query: {query}")
        
        # Apply filters to cached results
        filtered_products = apply_filters(products, category, min_price, max_price, selected_retailers)
        
        return render_template('search_results.html', 
                             products=filtered_products, 
                             query=query, 
                             last_updated=last_updated)
    
    try:
        # Scrape all sites for products using modular scrapers
        logger.info(f"Starting search for query: {query}")
        
        # Use modular scraper manager directly
        manager = ScraperManager()
        products_per_site = max(1, 20 // len(manager.get_available_sites()))
        products = manager.scrape_all_sites(query, products_per_site)[:20]
        
        if products:
            # Store products in database with currency conversion
            for product_data in products:
                # Convert price to GHS if needed
                original_price = product_data['price']
                ghs_price = get_numeric_price(original_price)
                
                # Check if product already exists
                product = Product.query.get(product_data['id'])
                if not product:
                    product = Product()
                    product.id = product_data['id']
                    product.name = product_data['title']
                    product.image_url = product_data['image_url']
                    product.description = ""
                    db.session.add(product)
                
                # Get or create source
                source = Source.query.filter_by(name=product_data['source']).first()
                if not source:
                    source = Source()
                    source.name = product_data['source']
                    source.url = ""
                    source.logo = product_data['source_key']
                    source.active = True
                    db.session.add(source)
                    db.session.flush()  # To get the ID
                
                # Check if price entry already exists
                existing_entry = PriceEntry.query.filter_by(
                    product_id=product_data['id'],
                    source_id=source.id
                ).first()
                
                if existing_entry:
                    # Update existing price entry with converted price
                    existing_entry.price = ghs_price
                    existing_entry.url = product_data.get('url', '')
                    existing_entry.last_updated = datetime.now()
                else:
                    # Create new price entry with converted price
                    price_entry = PriceEntry()
                    price_entry.product_id = product_data['id']
                    price_entry.source_id = source.id
                    price_entry.price = ghs_price
                    price_entry.url = product_data.get('url', '')
                    price_entry.last_updated = datetime.now()
                    db.session.add(price_entry)
                
                # Update product_data for display with converted price
                product_data['price'] = ghs_price
                product_data['formatted_price'] = format_price_with_currency(original_price, product_data['source_key'])
            
            db.session.commit()
            logger.info(f"Stored {len(products)} products in database")
            
            # Apply filters to results
            filtered_products = apply_filters(products, category, min_price, max_price, selected_retailers)
            
            # Cache the filtered results
            cache_results(cache_key, filtered_products)
            
            return render_template('search_results.html', 
                                 products=filtered_products, 
                                 query=query, 
                                 last_updated=datetime.now(),
                                 image_analysis=image_analysis)
        else:
            flash('No products found for your search', 'info')
            return render_template('search_results.html', 
                                 products=[], 
                                 query=query,
                                 image_analysis=image_analysis)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        flash('An error occurred while searching. Please try again.', 'error')
        return render_template('error.html', error=str(e))

@app.route('/product/<product_id>')
def product_details(product_id):
    """Show detailed view of a product with price comparison"""
    product = Product.query.get_or_404(product_id)
    price_entries = PriceEntry.query.filter_by(product_id=product_id).order_by(PriceEntry.price).all()
    best_price = product.get_best_price()
    
    return render_template('product_details.html', 
                         product=product, 
                         price_entries=price_entries, 
                         best_price=best_price)

@app.route('/refresh/<product_id>')
def refresh_product(product_id):
    """Refresh price information for a specific product"""
    product = Product.query.get_or_404(product_id)
    
    try:
        # Use the product name as query to refresh prices
        query = product.name
        
        # Invalidate cache for this query
        invalidate_cache(query)
        
        # Scrape fresh data using modular scrapers
        manager = ScraperManager()
        products_per_site = max(1, 20 // len(manager.get_available_sites()))
        products = manager.scrape_all_sites(query, products_per_site)[:20]
        
        # Update price entries for this product
        for product_data in products:
            if product_data['id'] == product_id:
                source = Source.query.filter_by(name=product_data['source']).first()
                if source:
                    existing_entry = PriceEntry.query.filter_by(
                        product_id=product_id,
                        source_id=source.id
                    ).first()
                    
                    if existing_entry:
                        existing_entry.price = product_data['price']
                        existing_entry.url = product_data['url']
                        existing_entry.last_updated = datetime.now()
                    else:
                        price_entry = PriceEntry()
                        price_entry.product_id = product_id
                        price_entry.source_id = source.id
                        price_entry.price = product_data['price']
                        price_entry.url = product_data['url']
                        price_entry.last_updated = datetime.now()
                        db.session.add(price_entry)
        
        db.session.commit()
        flash('Product prices have been refreshed', 'success')
        
    except Exception as e:
        logger.error(f"Error refreshing product {product_id}: {str(e)}")
        flash('Error refreshing product prices', 'error')
    
    return redirect(url_for('product_details', product_id=product_id))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error="Internal server error"), 500

@app.route('/api/search-progress')
def search_progress():
    """AJAX endpoint to check scraping progress"""
    from flask import session
    progress = session.get('search_progress', {'status': 'idle', 'sites_completed': 0, 'total_sites': 0})
    return jsonify(progress)


@app.route('/api/search-async', methods=['POST'])
def search_async():
    """AJAX endpoint for asynchronous search with progress tracking"""
    from flask import session
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Initialize progress tracking
    session['search_progress'] = {
        'status': 'searching',
        'sites_completed': 0,
        'total_sites': 7,
        'current_site': 'Starting search...'
    }
    
    try:
        # Start scraping
        logger.info(f"Starting async search for query: {query}")
        
        # Use modular scraper manager directly
        manager = ScraperManager()
        products_per_site = max(1, 20 // len(manager.get_available_sites()))
        products = manager.scrape_all_sites(query, products_per_site)[:20]
        
        # Update progress - completed
        session['search_progress'] = {
            'status': 'completed',
            'sites_completed': 7,
            'total_sites': 7,
            'current_site': 'Search completed'
        }
        
        return jsonify({
            'success': True,
            'products_found': len(products),
            'redirect_url': url_for('search', query=query)
        })
        
    except Exception as e:
        session['search_progress'] = {
            'status': 'error',
            'sites_completed': 0,
            'total_sites': 7,
            'current_site': f'Error: {str(e)}'
        }
        return jsonify({'error': str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return render_template('error.html', error="An unexpected error occurred"), 500
