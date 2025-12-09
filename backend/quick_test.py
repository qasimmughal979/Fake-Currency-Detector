#!/usr/bin/env python3
"""
Quick Gemini API Test - Lists available models and tests basic functionality
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"✓ API Key loaded: {api_key[:10]}...{api_key[-4:]}")

# Configure Gemini
genai.configure(api_key=api_key)

# List available models
print("\n📋 Available Models:")
print("=" * 60)
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  • {model.name}")
            print(f"    Description: {model.description[:80]}...")
            print()
except Exception as e:
    print(f"❌ Error listing models: {e}")
    exit(1)

print("\n🧪 Testing API with first available vision model...")
print("=" * 60)

# Try to find and use a vision model
try:
    vision_models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name.replace('models/', '')
            if 'vision' in model_name.lower() or 'pro' in model_name.lower():
                vision_models.append(model_name)
    
    if vision_models:
        test_model = vision_models[0]
        print(f"Using model: {test_model}")
        
        model = genai.GenerativeModel(test_model)
        response = model.generate_content("Say 'Hello from Gemini!'")
        
        print(f"\n✅ SUCCESS!")
        print(f"Response: {response.text}")
        print("\n🎉 Your Gemini API is working correctly!")
    else:
        print("❌ No suitable models found")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
