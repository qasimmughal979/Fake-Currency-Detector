"""
Gemini Vision-based Currency Verification Module
Uses Google's Gemini API to verify:
1. If the image contains a currency note
2. If it's specifically Pakistani currency
"""

import os
import google.generativeai as genai
from PIL import Image
import io

class GeminiCurrencyVerifier:
    def __init__(self, api_key=None):
        """
        Initialize Gemini verifier with API key
        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        # Use gemini-2.5-flash for better free tier limits
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini Vision verifier initialized (using gemini-2.5-flash)")
    
    def verify_currency(self, image_bytes):
        """
        Verify if image contains Pakistani currency
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            dict: {
                'is_currency': bool,
                'is_pakistani': bool,
                'confidence': str,
                'details': str
            }
        """
        try:
            # Convert bytes to PIL Image
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # Craft a precise prompt for verification
            prompt = """Analyze this image carefully and answer these questions:

1. Does this image contain a currency note (banknote/paper money)?
2. If yes, is it Pakistani currency (Pakistani Rupee)?

Respond in this exact JSON format:
{
    "is_currency": true/false,
    "is_pakistani": true/false,
    "confidence": "high/medium/low",
    "details": "brief description of what you see"
}

Be strict: only return is_currency=true if you clearly see a banknote."""

            # Generate response
            response = self.model.generate_content([prompt, pil_img])
            
            # Parse response
            result_text = response.text.strip()
            
            # Try to extract JSON from response
            import json
            # Remove markdown code blocks if present
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_text)
            
            print(f"Gemini Verification: Currency={result.get('is_currency')}, Pakistani={result.get('is_pakistani')}, Confidence={result.get('confidence')}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Gemini verification error: {str(e)}")
            # Return permissive result on error to not block analysis
            return {
                'is_currency': True,  # Assume yes on error
                'is_pakistani': True,
                'confidence': 'unknown',
                'details': f'Verification failed: {str(e)}'
            }
    
    def get_verification_message(self, verification_result):
        """
        Generate user-friendly message from verification result
        
        Args:
            verification_result: dict from verify_currency()
            
        Returns:
            tuple: (is_valid, message)
        """
        if not verification_result['is_currency']:
            return False, "No currency note detected in the image. Please upload a clear image of a banknote."
        
        if not verification_result['is_pakistani']:
            return False, "This appears to be a currency note, but not Pakistani Rupee. This system is designed for Pakistani currency only."
        
        # Valid Pakistani currency
        confidence = verification_result.get('confidence', 'unknown')
        return True, f"Pakistani currency detected (confidence: {confidence})"
