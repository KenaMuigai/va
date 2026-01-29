import ollama
import json
import os
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from weatherAPI import WeatherAPI
from CalendarAPI import CalendarAPI

SYSTEM_PROMPT = """
You are a concise, friendly assistant.
Rules:
- Answer simple factual questions directly.
- Use conversation history ONLY when necessary.
- Keep responses short and clear.
- Remember context for follow-up questions when appropriate.
- If you don't know the answer, say "I don't know".
"""

# --------------------
# WEATHER CONDITIONS
# --------------------
WEATHER_CONDITIONS = [
    "rain", "snow", "clear", "cloud", "cloudy",
    "mist", "fog", "sun", "sunny", "storm", "thunder"
]

MAX_CONTEXT_TURNS = 5  # Context expires after N turns

def normalize(text: str) -> str:
    return text.lower().strip()

def extract_location(text: str) -> Optional[str]:
    match = re.search(r"\bin\s+([A-Za-z\s]+)", text, re.IGNORECASE)
    if not match:
        return None
    loc = match.group(1)
    loc = re.sub(
        r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        "", loc, flags=re.IGNORECASE
    )
    loc = re.sub(
        r"\b(will|be|like|weather|forecast|on|at)\b",
        "", loc, flags=re.IGNORECASE
    )
    return loc.strip() or None

def extract_day(text: str) -> Optional[str]:
    t = normalize(text)
    if "today" in t:
        return "today"
    if "tomorrow" in t:
        return "tomorrow"
    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]:
        if d in t:
            return d
    return None

def resolve_day(day: str) -> str:
    if day == "today":
        return datetime.now().strftime("%A").lower()
    if day == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%A").lower()
    return day

def extract_weather_condition(text: str) -> Optional[str]:
    t = normalize(text)
    for cond in WEATHER_CONDITIONS:
        if cond in t:
            return cond
    return None

def condition_matches(requested: str, actual: str) -> bool:
    actual = normalize(actual)
    mapping = {
        "clear": ["clear"],
        "sun": ["sun", "clear"],
        "sunny": ["sun", "clear"],
        "cloud": ["cloud"],
        "cloudy": ["cloud"],
        "rain": ["rain", "shower"],
        "snow": ["snow"],
        "storm": ["storm", "thunder"],
        "thunder": ["thunder", "storm"],
        "mist": ["mist", "fog"],
        "fog": ["fog", "mist"]
    }
    return any(k in actual for k in mapping.get(requested, []))

def is_weather_query(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "weather", "forecast", "rain", "snow", "sun",
        "cloud", "temperature", "temp", "clear"
    ])

def is_temperature_query(text: str) -> bool:
    return any(k in normalize(text) for k in ["temperature", "temp"])

def is_calendar_query(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "calendar", "appointment", "meeting", "schedule", "event"
    ])

def is_add_event(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "add", "create", "schedule", "set up", "new appointment", "event"
    ])

def is_delete_event(text: str) -> bool:
    return any(k in normalize(text) for k in ["delete", "remove", "cancel"])

def is_update_event(text: str) -> bool:
    return any(k in normalize(text) for k in ["change", "update", "edit"])

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

        self.weather_api = WeatherAPI()
        self.calendar_api = CalendarAPI()

        # Last weather context for follow-ups
        self.last_weather_context = {"place": None, "day": None, "turn": 0}
        # Last calendar event for follow-ups
        self.last_calendar_event_id = None
        self.last_calendar_turn = 0

        self._load_memory()

    # --------------------
    # MEMORY FUNCTIONS
    # --------------------
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

    # --------------------
    # WEATHER CONTEXT FUNCTIONS
    # --------------------
    def _update_weather_context(self, place, day):
        self.last_weather_context = {
            "place": place,
            "day": day,
            "turn": len(self.history) // 2
        }

    def _get_weather_context(self):
        if (len(self.history)//2 - self.last_weather_context.get("turn",0)) > MAX_CONTEXT_TURNS:
            self.last_weather_context = {"place": None, "day": None, "turn":0}
            return None, None
        return self.last_weather_context.get("place"), self.last_weather_context.get("day")

    # --------------------
    # CALENDAR PARSING HELPERS
    # --------------------
    def _parse_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date from user text like '6th February' or '6 Feb'"""
        match = re.search(
            r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|march|april|may|june|july|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december))",
            text,
            re.IGNORECASE
        )
        if match:
            date_str = match.group(1)
            date_str_clean = re.sub(r"(st|nd|rd|th)", "", date_str, flags=re.IGNORECASE).strip()
            try:
                return datetime.strptime(date_str_clean + f" {datetime.now().year}", "%d %B %Y")
            except ValueError:
                try:
                    return datetime.strptime(date_str_clean + f" {datetime.now().year}", "%d %b %Y")
                except ValueError:
                    return None
        return None

    def _extract_title_from_text(self, text: str) -> str:
        """Extract title by removing date and keywords"""
        text_clean = text.lower()
        # Remove keywords
        text_clean = re.sub(r"\b(create|add|schedule|set up|event|appointment|titled|title|for)\b", "", text_clean)
        # Remove the date substring
        date_match = re.search(
            r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|march|april|may|june|july|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december))",
            text_clean,
            re.IGNORECASE
        )
        if date_match:
            text_clean = text_clean.replace(date_match.group(0), "")
        # Clean extra spaces
        return text_clean.strip().title() or "Untitled"

    # --------------------
    # GENERATE FUNCTION
    # --------------------
    def generate(self, user_text: str) -> str:
        user_norm = normalize(user_text)

        # Forget context command
        if user_norm == "/forget":
            self.last_weather_context = {"place": None, "day": None, "turn":0}
            self.last_calendar_event_id = None
            self.last_calendar_turn = 0
            return "Context forgotten."

        # --------------------
        # WEATHER INTENT
        # --------------------
        if is_weather_query(user_text):
            place = extract_location(user_text)
            day_key = extract_day(user_text)

            if not place or not day_key:
                last_place, last_day = self._get_weather_context()
                place = place or last_place or "Marburg"
                day_key = day_key or last_day or "today"

            resolved_day = resolve_day(day_key)
            self._update_weather_context(place, day_key)

            forecast = self.weather_api.get_forecast_day(place, resolved_day)
            if forecast.get("error"):
                return "I couldn't find that forecast."

            weather = forecast["weather"]
            tmin = forecast["temperature"]["min"]
            tmax = forecast["temperature"]["max"]

            requested_condition = extract_weather_condition(user_text)
            if requested_condition:
                yesno = "Yes" if condition_matches(requested_condition, weather) else "No"
                return (
                    f"{yesno}. The weather in {place} on {resolved_day} will be {weather}, "
                    f"with a temperature between {tmin}°C and {tmax}°C."
                )

            if is_temperature_query(user_text):
                return f"Today, the temperature in {place} will be between {tmin}°C and {tmax}°C."

            if day_key == "today":
                return f"Today, the weather in {place} will be {weather} with a temperature between {tmin}°C and {tmax}°C."
            return f"The weather in {place} on {resolved_day} will be {weather} with a temperature between {tmin}°C and {tmax}°C."

        # --------------------
        # CALENDAR INTENT
        # --------------------
        if is_calendar_query(user_text):
            events = self.calendar_api.list_events() or []

            # REMOVE expired events
            now = datetime.now()
            upcoming = []
            for e in events:
                try:
                    end_time = e.get("end_time")
                    if end_time.endswith("Z"):
                        dt = datetime.fromisoformat(end_time.replace("Z","+00:00")).astimezone()
                    else:
                        dt = datetime.fromisoformat(end_time)
                    if dt >= now:
                        upcoming.append(e)
                    else:
                        self.calendar_api.delete_event(e["id"])
                except Exception:
                    upcoming.append(e)
            events = upcoming

            # --------------------
            # ADD event
            # --------------------
            if is_add_event(user_text):
                title = self._extract_title_from_text(user_text)
                date_obj = self._parse_date_from_text(user_text)
                start_time = date_obj.isoformat() if date_obj else datetime.now().isoformat()
                end_time = (date_obj + timedelta(hours=1)).isoformat() if date_obj else (datetime.now() + timedelta(hours=1)).isoformat()

                event = self.calendar_api.create_event(
                    title=title,
                    description="Created via assistant",
                    start_time=start_time,
                    end_time=end_time,
                    location=self.facts.get("location", "Marburg")
                )
                self.last_calendar_event_id = event.get("id")
                self.last_calendar_turn = len(self.history)//2
                return f"Created appointment: {title}."

            # --------------------
            # DELETE event
            # --------------------
            if is_delete_event(user_text):
                # Check if user mentions a title
                title_match = re.search(r"titled\s+'?\"?([A-Za-z0-9\s]+)'?\"?", user_text, re.IGNORECASE)
                if title_match:
                    title_to_delete = title_match.group(1).strip().lower()
                    deleted = False
                    for e in events:
                        if e.get("title","").lower() == title_to_delete:
                            self.calendar_api.delete_event(e["id"])
                            deleted = True
                            break
                    if deleted:
                        return f"Deleted appointment titled '{title_to_delete}'."
                    else:
                        return f"No appointment found with title '{title_to_delete}'."
                elif self.last_calendar_event_id:
                    self.calendar_api.delete_event(self.last_calendar_event_id)
                    self.last_calendar_event_id = None
                    return "Deleted the previously created appointment."
                else:
                    return "You have no events to delete."

            # --------------------
            # UPDATE event
            # --------------------
            if is_update_event(user_text):
                if self.last_calendar_event_id:
                    new_loc = extract_location(user_text) or "Marburg"
                    self.calendar_api.update_event(self.last_calendar_event_id, location=new_loc)
                    return f"Updated the previously created appointment location to {new_loc}."
                elif events:
                    new_loc = extract_location(user_text) or "Marburg"
                    self.calendar_api.update_event(events[0]["id"], location=new_loc)
                    return f"Updated the event location to {new_loc}."
                else:
                    return "You have no events to update."

            # --------------------
            # LIST events
            # --------------------
            if "today" in user_text.lower():
                today_str = datetime.now().strftime("%Y-%m-%d")
                today_events = [e for e in events if e.get("start_time","").startswith(today_str)]
                if not today_events:
                    return "No calendar events for today."
                return "\n".join([f"You have an event '{e['title']}' on {datetime.fromisoformat(e['start_time']).strftime('%d %B')} at {e.get('location','TBA')}." for e in today_events])

            # Show all upcoming
            if not events:
                return "No calendar events found."
            return "\n".join([f"You have an event '{e['title']}' on {datetime.fromisoformat(e['start_time']).strftime('%d %B')} at {e.get('location','TBA')}." for e in events])

        # --------------------
        # FALLBACK
        # --------------------
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            )
            return response["message"]["content"].strip()
        except Exception:
            return "Sorry, I'm having trouble responding right now."

if __name__ == "__main__":
    llm = LLM()
    print("Local assistant running. Type 'exit' to quit. Type '/forget' to clear context.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("Assistant:", llm.generate(user_input))
