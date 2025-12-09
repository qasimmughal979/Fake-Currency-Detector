#!/usr/bin/env python3
"""
Gemini API Verification Script
Tests if Gemini API is properly configured and working
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_env_loading():
    """Test if .env file loads correctly"""
    print("=" * 60)
    print("TEST 1: Environment Variable Loading")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ python-dotenv installed and loaded")
    except ImportError:
        print("✗ python-dotenv not installed")
        print("  Run: pip3 install python-dotenv")
        return False
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✓ GEMINI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("✗ GEMINI_API_KEY not found in environment")
        print("  Make sure .env file exists in backend/ directory")
        return False

def test_google_ai_package():
    """Test if google-generativeai is installed"""
    print("\n" + "=" * 60)
    print("TEST 2: Google AI Package")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        print("✓ google-generativeai package installed")
        return True
    except ImportError as e:
        print(f"✗ google-generativeai not installed: {e}")
        print("  Run: pip3 install google-generativeai")
        return False

def test_gemini_connection():
    """Test actual connection to Gemini API"""
    print("\n" + "=" * 60)
    print("TEST 3: Gemini API Connection")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        import google.generativeai as genai
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("✗ No API key available")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')
        
        print("✓ Gemini API configured")
        print("  Testing with simple text prompt...")
        
        # Use text-only model for simple test
        text_model = genai.GenerativeModel('gemini-pro')
        response = text_model.generate_content("Say 'Hello, API is working!'")
        print(f"✓ API Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

def test_currency_verifier():
    """Test the currency verifier module"""
    print("\n" + "=" * 60)
    print("TEST 4: Currency Verifier Module")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from gemini_verifier import GeminiCurrencyVerifier
        
        verifier = GeminiCurrencyVerifier()
        print("✓ GeminiCurrencyVerifier initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Currency verifier failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n🔍 GEMINI API VERIFICATION SCRIPT")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Environment Loading", test_env_loading()))
    results.append(("Google AI Package", test_google_ai_package()))
    results.append(("API Connection", test_gemini_connection()))
    results.append(("Currency Verifier", test_currency_verifier()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your Gemini API is properly integrated and ready to use.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please fix the issues above before using the backend.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
