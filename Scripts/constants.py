import getpass
from openai import OpenAI
import time
from datetime import datetime, timezone

POS_LAT_SIRIUS = -22.804617
POS_LON_SIRIUS = -47.05297


def olog_constants():
    """
    Function that defines the constants used by the OLOG timestamp extraction model.

    Output:
    - SYSTEM_PROMPT: instruction prompt used to guide the language model in extracting
    event time intervals from log messages
    - BASE_URL: endpoint URL of the Gemini API service
    - API_KEY: user-provided API authentication key
    - MODEL_NAME: name of the language model used for timestamp extraction

    Returns:
    - tuple containing:
       * SYSTEM_PROMPT (str)
       * BASE_URL (str)
       * API_KEY (str)
       * MODEL_NAME (str)
    """

    SYSTEM_PROMPT = """
   You are a timestamp extraction engine.

   TASK
   Extract the event start and end times from the provided log.

   INPUT

   You will receive:
   - Reference date (YYYY-MM-DD)
   - Log text

   RULES

   1. Use ONLY information present in the log.
   2. The reference date must always be used when a time appears without a date.
   3. Assume timestamps are GMT-3 unless another timezone is explicitly specified.
   4. Convert all timestamps to UTC.
   5. Return timestamps in ISO 8601 format:
      YYYY-MM-DDTHH:MM:SS.000000Z

   6. If an explicit time interval is found:
      - first time -> from
      - second time -> to

   7. If only a single timestamp is found:
      - set "from" to that timestamp
      - set "to" to exactly 2 hours after "from"

   8. If multiple timestamps are mentioned but they do not explicitly define an interval:
      - use the earliest timestamp as "from"
      - use the latest timestamp as "to"

   9. If no timestamp can be determined:
      - from = null
      - to = null

   10. Never return explanations.
   11. Never return examples.
   12. Never repeat the input.
   13. Never return markdown.
   14. Return only a valid JSON object.
   15. All timestamps must always contain a complete ISO 8601 UTC datetime.

   OUTPUT FORMAT

   {
   "from": string | null,
   "to": string | null
   }
   """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    API_KEY = getpass.getpass("Insira a chave API: ")
    MODEL_NAME = "models/gemini-3.5-flash-lite"

    return SYSTEM_PROMPT, BASE_URL, API_KEY, MODEL_NAME
