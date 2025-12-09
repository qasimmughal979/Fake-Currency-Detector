# ✅ Gemini API Integration - Complete Setup Guide

## Current Status
✓ `.env` file created with your API key  
✓ `python-dotenv` installed  
✓ `google-generativeai` installed  
✓ API key verified and working  
✓ Using `gemini-2.5-flash` model (best for free tier)

## Files Created

### 1. `.env` - Environment Variables
```
GEMINI_API_KEY=AIzaSyC-VA-DpMOjE_Y4KGQHHZ0ZAu3a1AhzkXY
```
**⚠️ IMPORTANT**: Never commit this file to Git! It's already in `.gitignore`.

### 2. `gemini_verifier.py` - Main Verification Module
Handles currency detection and Pakistani currency verification.

### 3. `test_gemini.py` - Full Test Suite
Comprehensive verification script that tests all components.

### 4. `quick_test.py` - Quick API Test
Simple script to verify API key and list available models.

## How to Verify Integration

### Option 1: Quick Test (Recommended)
```bash
cd backend
python3 quick_test.py
```

This will:
- Load your API key from `.env`
- List all available Gemini models
- Test basic API connectivity

### Option 2: Full Test Suite
```bash
cd backend
python3 test_gemini.py
```

This runs comprehensive tests on all components.

### Option 3: Start the Backend
```bash
cd backend
python3 app.py
```

Look for this message:
```
✓ Gemini Vision verifier initialized (using gemini-2.5-flash)
```

## How It Works

1. **User uploads image** → Frontend sends to `/predict`
2. **Gemini Vision analyzes** → Checks if it's Pakistani currency
3. **If verified** → Proceeds to ResNet50 for fake/real classification
4. **If not verified** → Returns error message

## API Quota Information

Your free tier limits (gemini-2.5-flash):
- **15 requests per minute**
- **1,500 requests per day**
- **1 million tokens per day**

This is more than enough for development and testing!

## Troubleshooting

### "ModuleNotFoundError: No module named 'google'"
```bash
pip3 install google-generativeai
```

### "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip3 install python-dotenv
```

### "GEMINI_API_KEY not found"
Make sure:
1. `.env` file exists in `backend/` directory
2. File contains: `GEMINI_API_KEY=your-key-here`
3. No spaces around the `=` sign

### "Quota exceeded"
You've hit the daily/minute limit. Wait a bit or:
- Use `gemini-2.5-flash` instead of `gemini-2.5-pro`
- Check usage at: https://ai.dev/usage

## Next Steps

1. **Restart your backend** (if it's running):
   - Press `Ctrl+C` to stop
   - Run `python3 backend/app.py` again

2. **Test with the web app**:
   - Upload a Pakistani currency note
   - Should see: "Pakistani currency detected"
   - Then get Real/Fake classification

3. **Test with non-currency**:
   - Upload a random image (cat, car, etc.)
   - Should see: "No currency note detected"

4. **Test with non-Pakistani currency**:
   - Upload USD, EUR, etc.
   - Should see: "Not Pakistani Rupee"

## Security Best Practices

✓ API key is in `.env` file (not in code)  
✓ `.env` is in `.gitignore`  
✓ Never share your API key publicly  
✓ Rotate your key if accidentally exposed

## Support

- Gemini API Docs: https://ai.google.dev/docs
- Get API Key: https://makersuite.google.com/app/apikey
- Check Usage: https://ai.dev/usage
