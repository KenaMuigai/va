from CalendarAPI import CalendarAPI
from datetime import datetime, timedelta

api = CalendarAPI()

# 1. Create
event = api.create_event(
    title="Exam",
    description="Final exam",
    start_time="2026-01-30T10:00:00Z",
    end_time="2026-01-30T12:00:00Z",
    location="Room 101"
)
print("Created event:", event)

# 2. List
events = api.list_events()
print("All events:", events)

# 3. Update
if events:
    eid = events[-1]['id']
    updated = api.update_event(eid, location="Room 102")
    print("Updated event:", updated)

# 4. Get single
single = api.get_event(eid)
print("Single event:", single)

# 5. Delete
deleted = api.delete_event(eid)
print("Deleted event:", deleted)

# 6. List after deletion
events_after = api.list_events()
print("Events after deletion:", events_after)
