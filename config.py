import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Free tier = 20 requests/day PER MODEL; on quota (429) / overload (503, 504) / retired (404) the next model is tried.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.7-flash",
    ).split(",") if m.strip()
]

CATEGORIES = [
    "Electrical",
    "Plumbing / Water",
    "Sanitation",
    "Waste Management",
    "Infrastructure",
    "Security",
    "Cleaning / Housekeeping",
    "Furniture / Facilities",
    "Internet / Network",
    "Transport",
    "Other",
]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

DEPARTMENTS = {
    "Electrical": "Electrical Maintenance",
    "Plumbing / Water": "Plumbing / Facilities",
    "Sanitation": "Housekeeping / Sanitation",
    "Waste Management": "Waste Management",
    "Internet / Network": "IT Support",
    "Furniture / Facilities": "Facilities / Maintenance",
    "Security": "Security Office",
    "Cleaning / Housekeeping": "Housekeeping / Sanitation",
    "Infrastructure": "Civil / Maintenance",
    "Transport": "Transport Office",
    "Other": "General Maintenance",
}
