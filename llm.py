import ollama
import json
import os
import re
from typing import List, Dict, Optional
from datetime import datetime
import pytz

SYSTEM_PROMPT = """
You are a concise, friendly assistant.

Rules:
- Answer simple factual questions directly.
- Use conversation history ONLY when necessary.
- Do NOT restate past conversations unless asked.
- Keep responses short and clear.
- If user asks for a list, respond ONLY in list format (bullets or numbered).
- If you don't know the answer, say "I don't know" and do NOT hallucinate.
"""

# -----------------------------
# Utility functions
# -----------------------------

def normalize(text: str) -> str:
    return text.lower().strip()

def extract_name(text: str) -> Optional[str]:
    match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_location(text: str) -> Optional[str]:
    match = re.search(
        r"(?:i live in|i am from|i'm from)\s+([A-Z][a-zA-Z\s]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None

def needs_context(text: str) -> bool:
    keywords = [
        "earlier", "before", "again", "that",
        "we discussed", "you said", "last time",
        "continue", "remember"
    ]
    return any(k in normalize(text) for k in keywords)

def is_list_request(text: str) -> bool:
    return "list" in normalize(text) or "bullet" in normalize(text) or "suggestions as a list" in normalize(text)

def is_comparison_question(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "which is best", "best among", "which one", "top rated", "highest rated", "recommend one"
    ])

def is_confirmation(text: str) -> bool:
    return normalize(text) in {"yes", "yeah", "yep", "correct", "sure", "please do"}

def is_rejection(text: str) -> bool:
    return normalize(text) in {"no", "nope", "don't", "do not", "forget it"}

# -----------------------------
# Time / timezone utilities
# -----------------------------

TIMEZONE_MAP = {
    "germany": "Europe/Berlin",
    "marburg": "Europe/Berlin",
    "berlin": "Europe/Berlin",
    "france": "Europe/Paris",
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "usa": "America/New_York",
    "india": "Asia/Kolkata",
}

def get_timezone_for_location(location: str) -> Optional[str]:
    key = location.lower()
    return TIMEZONE_MAP.get(key)

def get_local_time(timezone_str: str) -> str:
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    return now.strftime("%I:%M %p")

# -----------------------------
# LLM Class
# -----------------------------

class LLM:
    def __init__(
        self,
        model="phi3:mini",
        history_file="conversation_history.json",
        facts_file="facts.json",
        max_exchanges=4,
    ):
        self.model = model
        self.history_file = history_file
        self.facts_file = facts_file
        self.max_exchanges = max_exchanges

        self.history: List[Dict] = []
        self.facts: Dict = {}
        self.pending_fact: Optional[Dict] = None

        # store last list output
        self.last_list: Optional[List[str]] = None

        self._load_memory()

    # -----------------------------
    # Memory handling
    # -----------------------------

    def _load_memory(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                self.history = []

        if os.path.exists(self.facts_file):
            try:
                with open(self.facts_file, "r") as f:
                    self.facts = json.load(f)
            except json.JSONDecodeError:
                self.facts = {}

    def _save_memory(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        with open(self.facts_file, "w") as f:
            json.dump(self.facts, f, indent=2)

    def _trim_history(self):
        exchanges = []
        buffer = []
        for msg in self.history:
            buffer.append(msg)
            if msg["role"] == "assistant":
                exchanges.append(buffer)
                buffer = []
        exchanges = exchanges[-self.max_exchanges:]
        self.history = [m for pair in exchanges for m in pair]

    # -----------------------------
    # Core generation
    # -----------------------------

    def generate(self, user_text: str) -> str:
        user_norm = normalize(user_text)

        # ---- Handle pending fact confirmation ----
        if self.pending_fact:
            if is_confirmation(user_text):
                key = self.pending_fact["key"]
                value = self.pending_fact["value"]
                self.facts[key] = value
                self.pending_fact = None
                self._save_memory()
                reply = f"Got it — I’ll remember that you {key.replace('_', ' ')} in {value}."
                self._append_exchange(user_text, reply)
                return reply

            if is_rejection(user_text):
                self.pending_fact = None
                reply = "Okay, I won’t remember that."
                self._append_exchange(user_text, reply)
                return reply

        # ---- Name (auto-store) ----
        name = extract_name(user_text)
        if name:
            self.facts["name"] = name
            self._save_memory()
            reply = f"Nice to meet you, {name}."
            self._append_exchange(user_text, reply)
            return reply

        # ---- Location (confirm before storing) ----
        location = extract_location(user_text)
        if location:
            self.pending_fact = {"key": "location", "value": location}
            reply = f"Got it — you live in {location}. Should I remember that?"
            self._append_exchange(user_text, reply)
            return reply

        # ---- Time queries (deterministic) ----
        if "time" in user_norm:
            if "in germany" in user_norm or "in berlin" in user_norm:
                tz = get_timezone_for_location("germany")
                return f"The current time in Germany is {get_local_time(tz)}."

            if "location" in self.facts:
                tz = get_timezone_for_location(self.facts["location"])
                if tz:
                    return f"The current time in {self.facts['location']} is {get_local_time(tz)}."

            return "I don't know your location yet. Please tell me where you live."

        # ---- Fact queries ----
        if "my name" in user_norm:
            return f"Your name is {self.facts['name']}." if "name" in self.facts else "I don't know your name yet."

        if "where do i live" in user_norm:
            return f"You live in {self.facts['location']}." if "location" in self.facts else "I don't know where you live yet."

        # ---- Build LLM input ----
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if needs_context(user_text):
            messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        # ---- If user asks comparison, inject last list ----
        if is_comparison_question(user_text) and self.last_list:
            list_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(self.last_list)])
            messages.append({
                "role": "user",
                "content": (
                    "Here are the options (from your previous list):\n"
                    f"{list_text}\n\n"
                    "Please pick the top-rated one ONLY from this list.\n"
                    "Do NOT use placeholders like [Insert].\n"
                    "If you are unsure, say 'I don't know'."
                )
            })

        # If user requests list, force it
        if is_list_request(user_text):
            messages.append({"role": "user", "content": "Please respond ONLY in list format."})

        # ---- Call Ollama safely ----
        try:
            response = ollama.chat(model=self.model, messages=messages)
            assistant_text = response["message"]["content"].strip()
            if not assistant_text:
                raise RuntimeError("Empty response")
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "Sorry, I'm having trouble responding right now. Please try again."

        # ---- Save last list if asked ----
        if is_list_request(user_text):
            items = [line.strip() for line in assistant_text.splitlines() if line.strip()]
            if len(items) >= 2:
                self.last_list = items

        self._append_exchange(user_text, assistant_text)
        return assistant_text

    def _append_exchange(self, user_text, assistant_text):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self._trim_history()
        self._save_memory()

    def reset_memory(self):
        self.history = []
        self.facts = {}
        self.pending_fact = None
        self.last_list = None
        self._save_memory()

# -----------------------------
# CLI Test
# -----------------------------

if __name__ == "__main__":
    llm = LLM()
    print("Local Ollama assistant running.")
    print("Commands: /reset | exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "/reset":
            llm.reset_memory()
            print("Assistant: Memory cleared.")
            continue

        reply = llm.generate(user_input)
        print("Assistant:", reply)
