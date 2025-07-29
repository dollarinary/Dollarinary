import json
import os
from trainable_ai import HybridDrawingAI

TRAINING_FILE = "training_data.json"

def refresh_with_new_prompts():
    ai = HybridDrawingAI()
    print("🔄 Refreshing TinyGPT2 Knowledge")

    if not os.path.exists(TRAINING_FILE):
        print("❌ No training data found.")
        return

    with open(TRAINING_FILE, "r") as f:
        data = json.load(f)

    for item in data:
        prompt = item["prompt"]
        response = item["response"]
        ai.learn(prompt, response)

    print("✅ Retraining complete. All prompts reprocessed.")

if __name__ == "__main__":
    refresh_with_new_prompts()
