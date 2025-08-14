import base64
import json
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_product_image(image_data):
    """
    Analyze uploaded image to extract product name and details using OpenAI Vision API
    
    Args:
        image_data: Base64 encoded image data or file path
    
    Returns:
        dict: Contains extracted product name, category, and search terms
    """
    try:
        # Validate OpenAI API key
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found")
            return {
                "product_name": "API Key Missing", 
                "category": "Error", 
                "brand": None,
                "features": [], 
                "search_terms": [], 
                "confidence": 0.0,
                "error": "OpenAI API key is required for image analysis"
            }
        
        # Handle different types of image_data input
        if isinstance(image_data, str):
            if image_data.startswith('data:'):
                # Extract base64 data from data URL
                base64_image = image_data.split(',')[1]
            elif len(image_data) > 500 and not image_data.startswith('/'):
                # Assume this is already base64 encoded data
                base64_image = image_data
            else:
                # Assume this is a file path
                try:
                    with open(image_data, 'rb') as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                except (FileNotFoundError, OSError) as e:
                    logger.error(f"Failed to read image file: {e}")
                    return {
                        "product_name": "File Read Error",
                        "category": "Error",
                        "brand": None,
                        "features": [],
                        "search_terms": [],
                        "confidence": 0.0,
                        "error": f"Could not read image file: {str(e)}"
                    }
        else:
            base64_image = image_data

        # Validate base64 image data
        if not base64_image or len(base64_image) < 100:
            logger.error("Invalid or empty base64 image data")
            return {
                "product_name": "Invalid Image",
                "category": "Error",
                "brand": None,
                "features": [],
                "search_terms": [],
                "confidence": 0.0,
                "error": "Invalid or empty image data provided"
            }

        # Ensure the base64 string is clean (no newlines, spaces, etc.)
        base64_image = base64_image.replace('\n', '').replace('\r', '').replace(' ', '')

        # Create OpenAI request with proper error handling
        try:
            logger.info("Sending image to OpenAI for analysis")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a product identification expert. Analyze the image and extract:
1. Primary product name (main item in the image)
2. Product category 
3. Key features/specifications visible
4. Brand name if visible
5. Alternative search terms

Respond with JSON in this exact format:
{
    "product_name": "specific product name",
    "category": "product category",
    "brand": "brand name or null",
    "features": ["feature1", "feature2"],
    "search_terms": ["term1", "term2", "term3"],
    "confidence": 0.95
}"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify the main product in this image and provide search terms for finding similar products in online stores."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )
        except Exception as api_error:
            logger.error(f"OpenAI API request failed: {str(api_error)}")
            raise ValueError(f"Image analysis service unavailable: {str(api_error)}")
        
        content = response.choices[0].message.content
        if content and content.strip():
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response: {content}")
                raise ValueError(f"Invalid JSON response from OpenAI: {e}")
        else:
            raise ValueError("Empty response from OpenAI")
        
        # Validate and clean the response
        cleaned_result = {
            "product_name": result.get("product_name", "").strip(),
            "category": result.get("category", "").strip(),
            "brand": result.get("brand", "").strip() if result.get("brand") else None,
            "features": result.get("features", []),
            "search_terms": result.get("search_terms", []),
            "confidence": min(1.0, max(0.0, result.get("confidence", 0.8)))
        }
        
        # Ensure we have at least a product name
        if not cleaned_result["product_name"]:
            cleaned_result["product_name"] = "Unknown Product"
            cleaned_result["confidence"] = 0.3
        
        logger.info(f"Successfully analyzed image: {cleaned_result['product_name']}")
        return cleaned_result
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {
            "product_name": "Image Analysis Failed",
            "category": "Unknown",
            "brand": None,
            "features": [],
            "search_terms": [],
            "confidence": 0.0,
            "error": str(e)
        }


def generate_search_query(analysis_result):
    """
    Generate optimized search query from image analysis result
    
    Args:
        analysis_result: Result from analyze_product_image()
    
    Returns:
        str: Optimized search query for product scraping
    """
    if analysis_result.get("error"):
        return "product"
    
    # Start with product name
    search_parts = []
    
    # Add brand if available
    if analysis_result.get("brand"):
        search_parts.append(analysis_result["brand"])
    
    # Add product name
    if analysis_result.get("product_name"):
        search_parts.append(analysis_result["product_name"])
    
    # Create primary search query
    primary_query = " ".join(search_parts)
    
    # Fallback to search terms if primary query is too generic
    if len(primary_query.split()) < 2 and analysis_result.get("search_terms"):
        primary_query = " ".join(analysis_result["search_terms"][:2])
    
    # Clean and return
    return primary_query.strip() or "product"


def get_alternative_queries(analysis_result, max_queries=3):
    """
    Generate alternative search queries for better product matching
    
    Args:
        analysis_result: Result from analyze_product_image()
        max_queries: Maximum number of alternative queries to generate
    
    Returns:
        list: List of alternative search queries
    """
    if analysis_result.get("error"):
        return []
    
    alternatives = []
    
    # Query with category
    if analysis_result.get("category") and analysis_result.get("product_name"):
        category_query = f"{analysis_result['category']} {analysis_result['product_name']}"
        alternatives.append(category_query)
    
    # Query with features
    if analysis_result.get("features"):
        for feature in analysis_result["features"][:2]:
            if analysis_result.get("product_name"):
                feature_query = f"{analysis_result['product_name']} {feature}"
                alternatives.append(feature_query)
    
    # Use search terms
    if analysis_result.get("search_terms"):
        for term in analysis_result["search_terms"][:max_queries]:
            if term not in alternatives:
                alternatives.append(term)
    
    return alternatives[:max_queries]