from dataclasses import dataclass


@dataclass
class KeyEstimationResult:
    key: str
    explanation: str
