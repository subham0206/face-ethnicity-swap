import os
from dotenv import load_dotenv
import google.generativeai as genai

# Force reload environment variables
load_dotenv(override=True)

print("=" * 50)
print("GOOGLE API KEY DEBUGGING")
print("=" * 50)

# Direct access to env vars
env_key = os.environ.get("GOOGLE_API_KEY")

print(f"GOOGLE_API_KEY env var exists: {env_key is not None}")
if env_key:
    print(f"GOOGLE_API_KEY env var length: {len(env_key)}")
    print(f"GOOGLE_API_KEY first few chars: {env_key[:10]}..." if len(env_key) > 10 else f"GOOGLE_API_KEY: {env_key}")
    print("GOOGLE_API_KEY contains quotes: {0}".format('"' in env_key or "'" in env_key))

# Try to configure and use the API
print("\nAttempting to configure Google Generative AI API...")
try:
    if env_key:
        # Remove any quotes that might be causing issues
        if env_key.startswith('"') or env_key.startswith("'"):
            clean_key = env_key.strip("\"'")
            print("Stripped quotes from API key")
        else:
            clean_key = env_key
            
        genai.configure(api_key=clean_key)
        print("API configured successfully, attempting a simple test call...")
        
        # Try a simple model listing to verify API works
        try:
            models = genai.list_models()
            print("API call successful! Available models:")
            for model in models:
                print(f" - {model.name}")
        except Exception as e:
            print(f"API call failed with error: {str(e)}")
    else:
        print("No API key available to test")
except Exception as e:
    print(f"Error configuring Google API: {str(e)}")

print("=" * 50)
print("DEBUGGING SUGGESTIONS:")
print("1. Ensure your Google API key is valid and has access to Generative AI API")
print("2. Make sure there are no quotes or special characters in your .env file")
print("3. Check if your API key has been enabled for generativelanguage.googleapis.com")
print("4. Verify billing is enabled on your Google Cloud project")
print("=" * 50)