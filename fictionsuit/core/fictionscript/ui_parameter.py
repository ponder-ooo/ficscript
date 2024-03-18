from __future__ import annotations

import uuid

class UiParameter:
    def __init__(self, type, value):
        self.type = type
        self.value = value
        self.cache_id = uuid.uuid4()

    def sm_schematize(self, cache):
        cache[str(self.cache_id)] = self
        return {"schema": "ui_parameter", "type": self.type, "value": self.value, "cache_id": str(self.cache_id)}

class UiButton:
    def __init__(self, text, action):
        self.text = text
        self.action = action
        self.cache_id = uuid.uuid4()

    def sm_schematize(self, cache):
        cache[str(self.cache_id)] = self
        return {"schema": "ui_button", "text": self.text, "action": self.action, "cache_id": str(self.cache_id)}