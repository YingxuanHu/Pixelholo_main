import os
from src.llm.llm_service import LLMService

# --- CONFIG ---
# If you haven't set this in your OS, uncomment and set it here:
# os.environ["GROQ_API_KEY"] = "gsk_..." 
# --------------

print("1. Initializing Brain...")
try:
    brain = LLMService()

    print("2. Asking Question: 'Explain quantum physics in 2 sentences.'")
    print("--- STREAM START ---")

    for i, sentence in enumerate(brain.stream_response("Explain quantum physics in 2 sentences.")):
        print(f"Sentence {i+1}: [{sentence}]")

    print("--- STREAM END ---")

except Exception as e:
    print(f"Test Failed: {e}")
