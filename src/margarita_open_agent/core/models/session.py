from dataclasses import dataclass


@dataclass
class SessionStartedMetadata:
    id: str
    model_id: str
