from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env into environment

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not found in .env file")
