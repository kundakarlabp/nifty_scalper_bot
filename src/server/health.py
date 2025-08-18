from __future__ import annotations
from flask import Flask, jsonify
from typing import Callable, Dict, Any

app = Flask(__name__)
status_callback: Callable[[], Dict[str, Any]] | None = None

@app.get("/health")
def health():
    if status_callback:
        status = status_callback()
    else:
        status = {"status": "ok", "message": "Status callback not configured."}
    return jsonify(status), 200

def run(callback: Callable[[], Dict[str, Any]] | None = None):
    global status_callback
    status_callback = callback
    app.run(host="0.0.0.0", port=8000)