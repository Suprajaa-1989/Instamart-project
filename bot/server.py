"""Tiny web server (standard library only): phone login, chat API and the page (bot/index.html)."""
import json
import os
import re
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bot import chat, data

PORT = int(os.environ.get("PORT", 8000))
PHONE_TO_CID, SESSIONS = data.load_phones(), {}                    # token -> customer id
PAGE = open("bot/index.html", "rb").read()                         # read once at start-up


def login(phone):
    digits = re.sub(r"\D", "", phone)[-10:]                        # accepts +91 and spaces
    cid = PHONE_TO_CID.get(digits)
    if cid is None:
        return {"ok": False}
    token = uuid.uuid4().hex
    SESSIONS[token] = cid
    rows, total = data.my_orders(cid, 5), len(data.my_orders(cid))
    return {"ok": True, "token": token, **data.customer_info(cid), "buttons": chat.MENU,
            "cards": [chat.orders_card(rows, f"Your recent orders (latest {len(rows)} of {total})")] if rows else [],
            "reply": "Hi! I found your account. Here are your recent orders - tap any order for details, or choose \u201CAll my orders\u201D to see all " + f"{total}. What do you need help with?"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        if self.path == "/api/demo":
            return self._json({"phones": list(PHONE_TO_CID)[:3]})
        self._send(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/login":
                return self._json(login(str(body.get("phone", ""))))
            cid = SESSIONS.get(str(body.get("token", "")))
            if cid is None:
                return self._json(chat.R("Session expired. Please sign in again.", buttons=[]), 401)
            self._json(chat.dispatch(cid, str(body.get("message", "")).strip()))
        except Exception as e:
            self._json(chat.R(f"Sorry, something went wrong ({type(e).__name__}). A human agent will follow up.", buttons=[]), 500)

    def log_message(self, *args):
        pass


def main():
    url = f"http://127.0.0.1:{PORT}"
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Instamart support chat running at {url}  (Ctrl+C to stop)")
    print("Demo phone numbers: " + ", ".join(f"{p} ({c})" for p, c in list(PHONE_TO_CID.items())[:3]))
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
