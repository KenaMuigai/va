import requests

class CalendarAPIError(Exception):
    pass

class CalendarAPI:
    def __init__(self, base_url="https://api.responsible-nlp.net/calendar.php", timeout=10, calenderid=54):
        self.base_url = base_url
        self.timeout = timeout
        self.calenderid = calenderid  # API expects "calenderid"
        self.headers = {"Content-Type": "application/json"}

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}
        except requests.RequestException as e:
            raise CalendarAPIError(f"Request failed: {e}")
        except ValueError:
            raise CalendarAPIError("Invalid JSON response")

    # -------------------- CREATE EVENT --------------------
    def create_event(self, title, description, start_time, end_time, location):
        payload = {
            "title": title,
            "description": description,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
        }
        response = requests.post(
            self.base_url,
            params={"calenderid": self.calenderid},  # pass as query param
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # -------------------- LIST EVENTS --------------------
    def list_events(self):
        response = requests.get(
            self.base_url,
            params={"calenderid": self.calenderid},
            timeout=self.timeout,
        )
        data = self._handle_response(response)
        if isinstance(data, dict) and "events" in data:
            return data["events"]
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
        return []

    # -------------------- GET SINGLE EVENT --------------------
    def get_event(self, event_id):
        response = requests.get(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # -------------------- UPDATE EVENT --------------------
    def update_event(self, event_id, **updates):
        response = requests.put(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            json=updates,
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # -------------------- DELETE EVENT --------------------
    def delete_event(self, event_id):
        response = requests.delete(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # -------------------- FORMAT EVENTS --------------------
    def event_to_text(self, event):
        if not isinstance(event, dict):
            return "Invalid event."
        return (
            f"Event #{event.get('id', '?')}\n"
            f"Title: {event.get('title', 'N/A')}\n"
            f"Description: {event.get('description', 'N/A')}\n"
            f"Start: {event.get('start_time', 'N/A')}\n"
            f"End: {event.get('end_time', 'N/A')}\n"
            f"Location: {event.get('location', 'N/A')}"
        )

    def events_to_text(self, events):
        if not events:
            return "No calendar events found."
        lines = ["Calendar events:", "-"*40]
        for event in events:
            if isinstance(event, dict):
                lines.append(self.event_to_text(event))
                lines.append("-"*40)
        return "\n".join(lines)
