"""
test_setup.py — Stage 0 connectivity check.

Loads OPENAI_API_KEY from .env, makes one minimal API call
(list available models), and prints the raw response so you
can confirm the key works before proceeding.
"""

import sys
from dotenv import load_dotenv
import os

# Load .env from the project root (one level up from src/ if run from there,
# or the current directory if run from the project root).
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found. Did you create a .env file?")
    sys.exit(1)

print(f"API key loaded: {api_key[:8]}...{api_key[-4:]}  (first 8 / last 4 chars shown)")

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

client = OpenAI(api_key=api_key)

print("\nCalling OpenAI API — listing available models ...\n")
models = client.models.list()

# Print first 10 model IDs so the output stays readable
model_ids = [m.id for m in models.data]
print(f"Total models returned: {len(model_ids)}")
print("First 10 model IDs:")
for mid in sorted(model_ids)[:10]:
    print(f"  {mid}")

print("\n✅ Setup check PASSED — OpenAI API is reachable and key is valid.")
