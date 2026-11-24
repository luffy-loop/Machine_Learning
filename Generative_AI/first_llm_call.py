
import os
from dotenv import load_dotenv
from google import genai

# Load Gemini API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("Gemini API key missing. Check .env")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

print("Connecting to Gemini AI...")

# Generate AI response
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="what is the capital of korea?"
)

print("\n===== GEMINI AI RESPONSE =====\n")
print(response.text)
