"""
Events & Hospitality AI Assistant
A small, explainable RAG prototype with strict event-level isolation.
Run: python event_assistant.py
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "events.json"
LOG_FILE = ROOT / "logs" / "conversations.jsonl"

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"disregard (all )?(previous|prior) instructions",
    r"reveal .*other event",
    r"give me .*other event",
    r"show me .*other event",
]
DISSATISFACTION_TERMS = {
    "unhappy", "dissatisfied", "angry", "frustrated", "terrible",
    "poor service", "complaint", "not happy", "disappointed"
}
VIP_TERMS = {"vip", "vip sponsor", "sponsor"}

class MockBookingAPI:
    """Mock booking API supplied by the assessment."""
    @staticmethod
    def get_booking_status(event_id: str, booking_id: str) -> str:
        return f"Booking {booking_id} for {event_id} is confirmed."


class EventKnowledgeBase:
    """Builds one TF-IDF index per event; retrieval never searches other events."""
    def __init__(self, data_file: Path = DATA_FILE):
        with data_file.open("r", encoding="utf-8") as f:
            self.events = json.load(f)
        self.indexes: dict[str, tuple[TfidfVectorizer, Any, list[dict]]] = {}
        for event_id, event in self.events.items():
            docs = event["documents"]
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            matrix = vectorizer.fit_transform([d["text"] for d in docs])
            self.indexes[event_id] = (vectorizer, matrix, docs)

    def event_name(self, event_id: str) -> str | None:
        event = self.events.get(event_id)
        return event["name"] if event else None

    def retrieve(self, event_id: str, query: str, threshold: float = 0.12):
        if event_id not in self.indexes:
            return None
        vectorizer, matrix, docs = self.indexes[event_id]
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()
        best_index = int(scores.argmax())
        score = float(scores[best_index])
        if score < threshold:
            return None
        return {"text": docs[best_index]["text"], "score": score,
                "document_id": docs[best_index]["id"]}


class ConversationLogger:
    def __init__(self, path: Path = LOG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict):
        safe_record = dict(record)
        safe_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


class EventAssistant:
    def __init__(self, kb: EventKnowledgeBase | None = None,
                 booking_api=MockBookingAPI, logger: ConversationLogger | None = None):
        self.kb = kb or EventKnowledgeBase()
        self.booking_api = booking_api
        self.logger = logger or ConversationLogger()
        self.sessions: dict[str, dict] = {}

    @staticmethod
    def _is_injection(text: str) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in INJECTION_PATTERNS)

    @staticmethod
    def _is_vip(text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in VIP_TERMS)

    @staticmethod
    def _is_dissatisfied(text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in DISSATISFACTION_TERMS)

    @staticmethod
    def _booking_id(text: str) -> str | None:
        # Accept formats such as "booking ID ABC123" or "booking #12345".
        match = re.search(
            r"booking\s*(?:id|number|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9_-]{2,})",
            text, re.IGNORECASE
        )
        if not match:
            return None
        candidate = match.group(1)
        if candidate.lower() in {"status", "details", "confirmation"}:
            return None
        return candidate

    def start_session(self, event_id: str, session_id: str | None = None) -> str:
        if event_id not in self.kb.events:
            raise ValueError("Unknown event_id. Use event_a or event_b.")
        session_id = session_id or str(uuid.uuid4())
        self.sessions[session_id] = {"event_id": event_id, "history": []}
        return session_id

    def reply(self, session_id: str, user_message: str) -> dict:
        if session_id not in self.sessions:
            raise ValueError("Session not found. Start a session with an event_id first.")

        session = self.sessions[session_id]
        event_id = session["event_id"]
        event_name = self.kb.event_name(event_id)
        text = user_message.strip()
        lower = text.lower()
        escalation_reasons = []

        # A session's event scope is immutable. Never accept an event switch in chat.
        mentions_other_event = (
            ("other event" in lower) or
            ("event a" in lower and event_id != "event_a") or
            ("event b" in lower and event_id != "event_b") or
            ("global tech summit" in lower and event_id != "event_a") or
            ("hospitality leadership forum" in lower and event_id != "event_b")
        )

        if self._is_vip(text):
            escalation_reasons.append("VIP user")
        if self._is_dissatisfied(text):
            escalation_reasons.append("User dissatisfaction")

        if self._is_injection(text) or mentions_other_event:
            answer = (
                f"I can only provide information for your selected event, {event_name}. "
                "I cannot access or disclose information belonging to another event. "
                "If you need help with a different event, please start a new session "
                "using that event's authorized event ID."
            )
            status = "blocked_cross_event_or_injection"
            source = None
        elif re.search(r"\b(booking|reservation)\b", lower) and re.search(r"\b(status|confirmed|confirmation)\b", lower):
            booking_id = self._booking_id(text)
            if booking_id:
                answer = self.booking_api.get_booking_status(event_id, booking_id)
                status = "booking_api_called"
                source = "mock_booking_api"
            else:
                answer = "Please share your booking ID so I can check its status for this event."
                status = "needs_booking_id"
                source = None
        else:
            result = self.kb.retrieve(event_id, text)
            if result:
                answer = result["text"]
                status = "answered_from_rag"
                source = result["document_id"]
            else:
                answer = (
                    f"I don't have verified information about that in the "
                    f"{event_name} knowledge base. I don't want to guess. "
                    "Please contact the event support team for confirmation."
                )
                status = "unknown_safe_fallback"
                source = None

        if escalation_reasons:
            answer += (
                "\n\nESCALATION: Your message has been flagged for the event support "
                f"team ({', '.join(escalation_reasons)})."
            )

        session["history"].append({"user": text, "assistant": answer})
        self.logger.write({
            "session_id": session_id,
            "event_id": event_id,
            "user_message": text,
            "assistant_message": answer,
            "status": status,
            "retrieved_source": source,
            "escalation": bool(escalation_reasons),
            "escalation_reasons": escalation_reasons
        })
        return {
            "answer": answer,
            "status": status,
            "event_id": event_id,
            "retrieved_source": source,
            "escalation": bool(escalation_reasons),
            "escalation_reasons": escalation_reasons
        }


def main():
    assistant = EventAssistant()
    print("\nEvents & Hospitality AI Assistant")
    print("Available events: event_a (Global Tech Summit), event_b (Hospitality Leadership Forum)")
    event_id = input("Select event ID: ").strip().lower()
    try:
        session_id = assistant.start_session(event_id)
    except ValueError as exc:
        print(exc)
        return
    print("Session started. Type 'exit' to quit.\n")
    while True:
        message = input("You: ").strip()
        if message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            result = assistant.reply(session_id, message)
            print(f"Assistant: {result['answer']}\n")
        except Exception as exc:
            print(f"Unable to process request: {exc}")


if __name__ == "__main__":
    main()
