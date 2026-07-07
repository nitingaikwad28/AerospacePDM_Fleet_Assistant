# chat_web.py
# Minimal Flask web chat UI for the Fleet Assistant - one page, one JSON endpoint.
# Run from inside fleet_assistant/:
#   python chat_web.py
# then open http://localhost:5000

from flask import Flask, request, jsonify, send_from_directory

import llm_client
import agent

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "chat.html")


@app.route("/api/status")
def status():
    return jsonify({"llm_available": llm_client.is_available()})


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"text": "Ask me something about the fleet.", "source": "clarification"})
    result = agent.answer(message)
    return jsonify(result)


if __name__ == "__main__":
    print("FleetGuard AI Fleet Assistant - open http://localhost:5000 in your browser")
    # threaded=True so the UI's /api/status check can't block a slower /api/chat call
    # (or vice versa) behind it - Flask's dev server is single-threaded by default.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
