"""Live triage helpers for the customer chat app (same rules as Part C, one ticket at a time)."""
import json
import random
import re
import threading
import time

import mock_llm

LOCK = threading.Lock()          # the mock keeps global counters, so serialise model calls
MAX_INPUT_TOKENS, MAX_RETRIES, BASE_DELAY = 400, 3, 0.05
CATS = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
FEW_SHOT = ("You are a support-ticket classifier. Your task is to CLASSIFY the customer ticket into exactly one "
            "category from this list: billing, delivery, technical, fraud. "
            'Reply ONLY with JSON of the form {"category": "...", "reason": "..."} and nothing else.\n'
            "EXAMPLES:\n"
            'Example 1 - Ticket: I was charged twice and need the extra amount refunded.\nAnswer: {"category": "billing", "reason": "duplicate charge"}\n'
            'Example 2 - Ticket: My order has not arrived and the rider is not picking up calls.\nAnswer: {"category": "delivery", "reason": "order not delivered"}\n'
            'Example 3 - Ticket: The app crashes every time I open the cart.\nAnswer: {"category": "technical", "reason": "app crash"}\n'
            'Example 4 - Ticket: There is an unauthorized transaction on my card that I never made.\nAnswer: {"category": "fraud", "reason": "unauthorized transaction"}')
random.seed(42)


def redact(text):
    for _, pat, ph in PATTERNS:
        text = pat.sub(ph, text)
    return text


def validate(content):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        cat = obj.get("category")
        return ("classified_json", cat) if cat in CATS else ("flagged_invalid_output", "")
    found = set(re.findall(r"[a-z]+", str(content).lower())) & set(CATS)
    return ("classified_keyword", found.pop()) if len(found) == 1 else ("flagged_invalid_output", "")


def classify(raw_text):
    """Returns (status, category). Guardrails: empty/long -> human; PII redacted before the model sees it."""
    if not raw_text.strip():
        return "skipped_empty", ""
    if mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
        return "flagged_long", ""
    safe = redact(raw_text)
    with LOCK:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = mock_llm.chat([{"role": "system", "content": FEW_SHOT}, {"role": "user", "content": safe}])
                return validate(resp.choices[0].message.content)
            except mock_llm.TransientLLMError:
                if attempt == MAX_RETRIES:
                    return "flagged_invalid_output", ""
                time.sleep(BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15))


def route(status, category, is_premium):
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "human_requested"}:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"
