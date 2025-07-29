import os
import json
import uuid
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

TRAINING_FILE = "training_data.json"
DEFAULT_MODEL = "distilgpt2"

class HybridDrawingAI:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        self.model.eval()
        self.training_data = self.load_training_data()

    def load_training_data(self) -> List[Dict]:
        if not os.path.exists(TRAINING_FILE):
            with open(TRAINING_FILE, 'w') as f:
                json.dump([], f)
        with open(TRAINING_FILE, 'r') as f:
            return json.load(f)

    def save_training_data(self):
        with open(TRAINING_FILE, 'w') as f:
            json.dump(self.training_data, f, indent=2)

    def generate_local(self, prompt: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    def suggest(self, prompt: str) -> str:
        best_match = None
        for item in self.training_data:
            if prompt.lower().strip() in item["prompt"].lower():
                best_match = item["response"]
                break

        if best_match:
            return best_match

        local_generated = self.generate_local(prompt)
        if local_generated:
            return local_generated

        return "No confident match. Try rephrasing."

    def learn(self, prompt: str, response: str, rating: int = 5, correction: str = None):
        self.training_data.append({
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "response": correction or response,
            "rating": rating
        })
        self.save_training_data()
