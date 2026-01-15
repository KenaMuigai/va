import ollama
import json
import os

SYSTEM_PROMPT = """
You are a concise and friendly assistant.
Answer questions clearly and politely.
Maintain context from previous conversation turns.
"""

class LLM:
    def __init__(self, model="phi3:mini", history_file="conversation_history.json", max_memory=10):
        """
        :param model: Ollama model name
        :param history_file: JSON file to persist conversation
        :param max_memory: max number of messages to keep for context
        """
        self.model = model
        self.history_file = history_file
        self.max_memory = max_memory
        self.history = []

        # Load existing history if available
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                print("Warning: history file corrupted. Starting fresh.")
                self.history = []

    def generate(self, user_text: str) -> str:
        # Add user input to history
        self.history.append({"role": "user", "content": user_text})

        # Trim history to max_memory
        if len(self.history) > self.max_memory:
            self.history = self.history[-self.max_memory:]

        # Prepare messages for Ollama chat API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        # Call Ollama
        response = ollama.chat(model=self.model, messages=messages)
        assistant_text = response["message"]["content"]

        # Add assistant reply to history
        self.history.append({"role": "assistant", "content": assistant_text})

        # Save history to JSON file
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        return assistant_text

    def reset_history(self):
        """Clear conversation memory and the JSON file"""
        self.history = []
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    llm = LLM()
    print("Phi-3 Mini with persistent memory running. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = llm.generate(user_input)
        print("Assistant:", reply)
