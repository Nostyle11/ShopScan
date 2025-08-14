import base64
import json
import os
import logging
from openai import OpenAI
from PIL import Image
import io

logger = logging.getLogger(__name__)

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def validate_image(image_data):
    """Validate and process image data"""
    try:
        # If it's a file-like object, read it
        if hasattr(image_data, 'read'):
            image_bytes = image_data.read()
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError("Invalid image data format")
        
        # Validate it's actually an image using PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
        
        # Check file size (max 20MB)
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError("Image file too large. Maximum size is 20MB.")
        
        # Convert to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise ValueError(f"Image processing failed: {str(e)}")


def analyze_product_image(image_file):
    """
    Analyze uploaded image to extract product information using OpenAI Vision API
    
    Args:
        image_file: File object from Flask request
    
    Returns:
        dict: Product analysis results
    """
    try:
        # Check if OpenAI client is available
        if not openai_client:
            return {
                "success": False,
                "error": "OpenAI API key not configured. Please check your environment variables.",
                "product_name": "Configuration Error",
                "category": "Error",
                "brand": None,
                "features": [],
                "search_terms": [],
                "confidence": 0.0
            }
        
        # Validate and process the image
        try:
            base64_image = validate_image(image_file)
            logger.info(f"Image validated successfully, size: {len(base64_image)} characters")
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "product_name": "Invalid Image",
                "category": "Error",
                "brand": None,
                "features": [],
                "search_terms": [],
                "confidence": 0.0
            }
        
        # Prepare the OpenAI Vision API request
        try:
            logger.info("Sending image to OpenAI Vision API for analysis")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert product identification system. Analyze the image and extract detailed product information.

Your task is to identify:
1. The main product name (be specific and accurate)
2. Product category (electronics, clothing, home, etc.)
3. Brand name if visible or identifiable
4. Key features and specifications visible in the image
5. Alternative search terms that would help find this product online

Return your analysis as a JSON object with this exact structure:
{
    "product_name": "specific product name",
    "category": "product category",
    "brand": "brand name or null if not identifiable",
    "features": ["feature1", "feature2", "feature3"],
    "search_terms": ["term1", "term2", "term3"],
    "confidence": 0.95
}

Be accurate and specific. If you cannot identify the product clearly, indicate lower confidence."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this product image and provide detailed identification information that would help someone find this exact product or similar products online."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1  # Low temperature for more consistent results
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from OpenAI")
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {content}")
                raise ValueError(f"Invalid response format from AI service: {str(e)}")
            
            # Validate and clean the response
            analysis_result = {
                "success": True,
                "product_name": result.get("product_name", "Unknown Product").strip(),
                "category": result.get("category", "Unknown").strip(),
                "brand": result.get("brand", "").strip() or None,
                "features": result.get("features", [])[:5],  # Limit to 5 features
                "search_terms": result.get("search_terms", [])[:5],  # Limit to 5 terms
                "confidence": max(0.0, min(1.0, result.get("confidence", 0.8)))
            }
            
            # Ensure we have at least a product name
            if not analysis_result["product_name"] or analysis_result["product_name"] == "Unknown Product":
                analysis_result["product_name"] = "Unidentified Product"
                analysis_result["confidence"] = 0.3
            
            logger.info(f"Successfully analyzed image: {analysis_result['product_name']} (confidence: {analysis_result['confidence']})")
            return analysis_result
            
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            return {
                "success": False,
                "error": f"AI analysis failed: {str(api_error)}",
                "product_name": "Analysis Failed",
                "category": "Error",
                "brand": None,
                "features": [],
                "search_terms": [],
                "confidence": 0.0
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in image analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "product_name": "System Error",
            "category": "Error",
            "brand": None,
            "features": [],
            "search_terms": [],
            "confidence": 0.0
        }


def generate_search_query(analysis_result):
    """
    Generate optimized search query from image analysis result
    
    Args:
        analysis_result: Result from analyze_product_image()
    
    Returns:
        str: Optimized search query for product scraping
    """
    if not analysis_result.get("success") or analysis_result.get("error"):
        return "product"
    
    # Build search query from analysis
    query_parts = []
    
    # Add brand if available
    if analysis_result.get("brand"):
        query_parts.append(analysis_result["brand"])
    
    # Add product name
    if analysis_result.get("product_name") and analysis_result["product_name"] not in ["Unknown Product", "Unidentified Product"]:
        query_parts.append(analysis_result["product_name"])
    
    # If we don't have enough terms, use search terms
    if len(query_parts) < 2 and analysis_result.get("search_terms"):
        query_parts.extend(analysis_result["search_terms"][:2])
    
    # Create final query
    search_query = " ".join(query_parts).strip()
    
    # Fallback to category if still empty
    if not search_query and analysis_result.get("category"):
        search_query = analysis_result["category"]
    
    # Final fallback
    if not search_query:
        search_query = "product"
    
    return search_query


def get_alternative_queries(analysis_result, max_queries=3):
    """
    Generate alternative search queries for better product matching
    
    Args:
        analysis_result: Result from analyze_product_image()
        max_queries: Maximum number of alternative queries to generate
    
    Returns:
        list: List of alternative search queries
    """
    if not analysis_result.get("success"):
        return []
    
    alternatives = []
    
    # Query with category and product name
    if analysis_result.get("category") and analysis_result.get("product_name"):
        category_query = f"{analysis_result['category']} {analysis_result['product_name']}"
        alternatives.append(category_query)
    
    # Queries with features
    if analysis_result.get("features"):
        for feature in analysis_result["features"][:2]:
            if analysis_result.get("product_name"):
                feature_query = f"{analysis_result['product_name']} {feature}"
                alternatives.append(feature_query)
    
    # Use search terms directly
    if analysis_result.get("search_terms"):
        for term in analysis_result["search_terms"]:
            if term not in alternatives and len(alternatives) < max_queries:
                alternatives.append(term)
    
    return alternatives[:max_queries]