import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors


class GeminiKeyEstimator:
    MODEL = "gemini-3.5-flash"

    def __init__(self):
        load_dotenv()
        api_keys = json.loads(os.getenv('GEMINI_API_KEYS'))
        self.clients = [genai.Client(api_key=api_key) for api_key in api_keys]
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                },
                required=["explanation", "key"],
            ),
        )

        prompt_path = Path(__file__).parent / 'prompt.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, chords: str) -> str:
        contents = self.prompt.replace("{CHORDS}", chords)
        num_clients = len(self.clients)

        for i in range(num_clients):
            while True:
                try:
                    response = self.clients[i].models.generate_content(
                        model=self.MODEL,
                        contents=contents,
                        config=self.config,
                    )
                    return response.text
                except errors.APIError as e:
                    if e.code == 503:
                        time.sleep(30)
                        continue
                    if e.code == 429 and i < num_clients - 1:
                        break
                    raise
