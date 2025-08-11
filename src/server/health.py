from __future__ import annotations
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

def run():
    app.run(host="0.0.0.0", port=8000)