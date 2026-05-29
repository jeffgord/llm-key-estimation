import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from result import KeyEstimationResult


class ClaudeKeyEstimator:
    MODEL = "claude-sonnet-4-6"

    _TOOL = {
        "name": "report_key",
        "description": "Report the predicted musical key",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["key", "explanation"],
        },
    }

    def __init__(self):
        load_dotenv()
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        prompt_path = Path(__file__).parent / 'prompt.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, chords: str, retries: int = 3) -> KeyEstimationResult:
        contents = self.prompt.replace("{CHORDS}", chords)

        last_exc: Exception = ValueError("No usable block in Claude response")
        for _ in range(retries):
            while True:
                try:
                    response = self.client.messages.create(
                        model=self.MODEL,
                        max_tokens=16000,
                        thinking={"type": "enabled", "budget_tokens": 15000},
                        tools=[self._TOOL],
                        tool_choice={"type": "auto"},
                        messages=[{"role": "user", "content": contents}],
                    )
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code == 529:
                        time.sleep(30)
                        continue
                    raise

            for block in response.content:
                if block.type == "tool_use" and block.name == "report_key":
                    return KeyEstimationResult(key=block.input['key'], explanation=block.input['explanation'])
            try:
                for block in response.content:
                    if block.type == "text":
                        data = json.loads(block.text)
                        return KeyEstimationResult(key=data['key'], explanation=data['explanation'])
            except (json.JSONDecodeError, KeyError) as e:
                last_exc = e

        raise last_exc
