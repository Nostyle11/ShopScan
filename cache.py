import logging
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Simple in-memory cache
# In a production app, you might use Redis or another caching system
CACHE = {}
CACHE_TTL = 3600  # Cache expiration time in seconds (1 hour)

def get_cached_results(query):
    """
    Retrieve cached search results for a query if they exist and aren't expired
    
    Returns a tuple of (products, last_updated) or None if no valid cache exists
    """
    normalized_query = query.lower().strip()
    
    if normalized_query in CACHE:
        cache_entry = CACHE[normalized_query]
        cache_time = cache_entry['timestamp']
        current_time = time.time()
        
        # Check if cache is still valid
        if current_time - cache_time < CACHE_TTL:
            logger.debug(f"Cache hit for query: {query}")
            return cache_entry['products'], cache_entry['last_updated']
        else:
            logger.debug(f"Cache expired for query: {query}")
            # Clean up expired cache entry
            del CACHE[normalized_query]
    
    logger.debug(f"Cache miss for query: {query}")
    return None

def cache_results(query, products):
    """Cache search results for a query"""
    normalized_query = query.lower().strip()
    
    CACHE[normalized_query] = {
        'products': products,
        'timestamp': time.time(),
        'last_updated': datetime.now()
    }
    
    logger.debug(f"Cached results for query: {query}")
    
    # Clean up cache if it's getting too large
    if len(CACHE) > 100:  # Arbitrary limit
        clean_cache()

def clean_cache():
    """Remove expired entries from the cache"""
    current_time = time.time()
    expired_keys = []
    
    for key, cache_entry in CACHE.items():
        if current_time - cache_entry['timestamp'] > CACHE_TTL:
            expired_keys.append(key)
    
    for key in expired_keys:
        del CACHE[key]
    
    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

def invalidate_cache(query=None):
    """
    Invalidate cache entries
    
    If query is provided, only that query's cache is invalidated.
    Otherwise, the entire cache is cleared.
    """
    if query:
        normalized_query = query.lower().strip()
        if normalized_query in CACHE:
            del CACHE[normalized_query]
            logger.debug(f"Invalidated cache for query: {query}")
    else:
        CACHE.clear()
        logger.debug("Cleared entire cache")
