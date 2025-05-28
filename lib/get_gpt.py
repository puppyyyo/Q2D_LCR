import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def init_openai_client(openai_api_key=None):
    """Initialize OpenAI client."""

    api_key = openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API Key. Please set it in .env or pass it explicitly.")

    client = OpenAI(api_key=api_key)

    return client

def get_gpt_response(client, prompt: str) -> str:
    """Fetch a response from GPT with retry mechanism (max 3 attempts)."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "你是一個專業的法律助理，擅長分析臺灣的判決書。且協助我進行案件情節的因子分析。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                n=1,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                return f"GPT API Error: {str(e)}"

    return None

def parse_json_response(response):
    """Parse the JSON response from GPT."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None
