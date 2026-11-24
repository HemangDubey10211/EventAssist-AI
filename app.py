
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "conversations.jsonl"

st.set_page_config(
    page_title="EventAssist AI",
    page_icon="🎫",
    layout="wide"
)


# --------------------------------------------------
# EVENT DATA
# --------------------------------------------------

EVENTS = {
    "Global Tech Summit": {
        "id": "event_a",
        "facts": [
            "Registration opens at 8:00 AM.",
            "Hall A is located on Level 1.",
            "VIP sponsors have access to the VIP lounge.",
            "Cancellation is allowed up to 72 hours before the event."
        ]
    },

    "Hospitality Leadership Forum": {
        "id": "event_b",
        "facts": [
            "Registration opens at 9:00 AM.",
            "Hall C is located on Level 3.",
            "There is no lounge access for attendees.",
            "Cancellation is allowed up to 48 hours before the event."
        ]
    }
}


# --------------------------------------------------
# --------------------------------------------------
# EVENT-SPECIFIC RAG KNOWLEDGE BASE
# --------------------------------------------------

class EventKnowledgeBase:

    def __init__(self):
        self.indexes = {}

        for event_name, event in EVENTS.items():
            documents = event["facts"]

            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(documents)

            self.indexes[event_name] = {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "documents": documents
            }

    def search(self, event_name, question):

        if event_name not in self.indexes:
            return None, 0

        index = self.indexes[event_name]

        query_vector = index["vectorizer"].transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            index["matrix"]
        )[0]

        best_index = scores.argmax()
        best_score = float(scores[best_index])

        if best_score < 0.12:
            return None, best_score

        return index["documents"][best_index], best_score

        index = self.indexes[event_name]

        query_vector = index["vectorizer"].transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            index["matrix"]
        )[0]

        best_index = scores.argmax()
        best_score = float(scores[best_index])

        if best_score < 0.12:
            return None, best_score

        return index["documents"][best_index], best_score


# --------------------------------------------------
# MOCK BOOKING API
# --------------------------------------------------

def get_booking_status(event_id, booking_id):

    return (
        f"Booking {booking_id} for {event_id} is confirmed."
    )


# --------------------------------------------------
# SECURITY
# --------------------------------------------------

def detect_injection(question):

    patterns = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"reveal.*other event",
        r"give me data of other event",
        r"show.*other event",
        r"bypass.*security",
        r"system prompt"
    ]

    return any(
        re.search(pattern, question.lower())
        for pattern in patterns
    )


def mentions_other_event(question, selected_event):

    other_events = [
        name for name in EVENTS
        if name != selected_event
    ]

    question_lower = question.lower()

    if any(
        event.lower() in question_lower
        for event in other_events
    ):
        return True

    if re.search(
        r"\b(other event|another event|different event)\b",
        question_lower
    ):
        return True

    return False


# --------------------------------------------------
# HUMAN ESCALATION
# --------------------------------------------------

def check_escalation(question):

    text = question.lower()

    vip = any(
        word in text
        for word in ["vip", "sponsor", "premium"]
    )

    unhappy = any(
        word in text
        for word in [
            "unhappy",
            "angry",
            "dissatisfied",
            "complaint",
            "poor service",
            "not satisfied",
            "frustrated"
        ]
    )

    if vip or unhappy:
        return True

    return False


# --------------------------------------------------
# BOOKING ID EXTRACTION
# --------------------------------------------------

def extract_booking_id(question):

    # First look for an explicit booking ID.
    match = re.search(
        r"\b(?:booking\s*(?:id|number)|reservation\s*(?:id|number))"
        r"\s*[:#-]?\s*([A-Za-z0-9_-]{4,})\b",
        question,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    # Also support messages such as: booking ABC123
    match = re.search(
        r"\bbooking\s+(?!status\b|details\b|information\b|id\b|number\b)"
        r"([A-Za-z0-9_-]{4,})\b",
        question,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# --------------------------------------------------
# CONVERSATION LOGGING
# --------------------------------------------------

def save_log(event_name, question, answer):

    record = {
        "timestamp": datetime.now().isoformat(),
        "event": event_name,
        "user_message": question,
        "assistant_response": answer
    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(json.dumps(record) + "\n")


# --------------------------------------------------
# MAIN AI ASSISTANT
# --------------------------------------------------

class EventAssistant:

    def __init__(self):
        self.knowledge_base = EventKnowledgeBase()

    def respond(self, event_name, question):

        # Security check
        if detect_injection(question):

            return (
                "I cannot follow instructions that attempt "
                "to bypass security or access another event's "
                "information. I can only help with the "
                "currently selected event."
            ), "Security blocked"

        # Event isolation check
        if mentions_other_event(question, event_name):

            return (
                "For security reasons, I can only provide "
                "information about the event currently "
                "selected in this session."
            ), "Event isolation"

        # Escalation check
        escalation = check_escalation(question)

        answer_parts = []

        if escalation:

            answer_parts.append(
                "I understand your concern. I am escalating "
                "this conversation to the support team "
                "for human assistance."
            )

        # Booking API
        if re.search(
            r"\b(booking|reservation)\b",
            question,
            re.IGNORECASE
        ):

            booking_id = extract_booking_id(question)

            if booking_id:

                booking_result = get_booking_status(
                    EVENTS[event_name]["id"],
                    booking_id
                )

                answer_parts.append(booking_result)

                return (
                    "\n\n".join(answer_parts),
                    "Escalated" if escalation else "Booking API"
                )

            answer_parts.append(
                "Please provide your booking ID so I can "
                "check its status."
            )

            return (
                "\n\n".join(answer_parts),
                "Escalated" if escalation else "Booking information"
            )

        
                # Stop knowledge retrieval after escalation
        if escalation:
            return "\n\n".join(answer_parts), "Escalated"
        # Event-specific knowledge retrieval
        document, score = self.knowledge_base.search(
            event_name,
            question
        )

        if document:

            answer_parts.append(document)

            answer_parts.append(
                f"Source: {event_name} event information."
            )

        else:

            answer_parts.append(
                "I do not have verified information about "
                "that question for this event. Please contact "
                "the event support team for assistance."
            )

        if escalation:
            status = "Escalated"

        elif document:
            status = "Knowledge base"

        else:
            status = "Unknown question"

        return "\n\n".join(answer_parts), status


# --------------------------------------------------
# INITIALIZE STREAMLIT SESSION
# --------------------------------------------------

if "assistant" not in st.session_state:

    st.session_state.assistant = EventAssistant()

if "messages" not in st.session_state:

    st.session_state.messages = []

if "current_event" not in st.session_state:

    st.session_state.current_event = "Global Tech Summit"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🎫 EventAssist AI")

    st.caption(
        "Enterprise Events & Hospitality Assistant"
    )

    st.divider()

    st.subheader("Select your event")

    selected_event = st.selectbox(
        "Active event",
        list(EVENTS.keys()),
        index=list(EVENTS.keys()).index(
            st.session_state.current_event
        )
    )

    if selected_event != st.session_state.current_event:

        st.session_state.current_event = selected_event
        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.subheader("Event information")

    for fact in EVENTS[selected_event]["facts"]:

        st.write("• " + fact)

    st.divider()

    st.subheader("Assistant status")

    st.success("Knowledge base active")
    st.success("Event isolation enabled")
    st.success("Conversation logging enabled")

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("🎫 EventAssist AI")

st.caption(
    "Your intelligent assistant for event information, "
    "booking support and attendee assistance."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Active Event",
        selected_event
    )

with col2:

    st.metric(
        "Knowledge Documents",
        len(EVENTS[selected_event]["facts"])
    )

with col3:

    st.metric(
        "Conversation Messages",
        len(st.session_state.messages)
    )

st.divider()

st.subheader("💬 Chat with your event assistant")

st.caption(
    f"Currently connected to: {selected_event}"
)


# --------------------------------------------------
# SUGGESTED QUESTIONS
# --------------------------------------------------

st.write("Try asking:")

suggestions = [
    "When does registration open?",
    "Where is the event hall?",
    "What is the cancellation policy?",
    "Do VIP sponsors get lounge access?"
]

suggestion_columns = st.columns(2)

for i, suggestion in enumerate(suggestions):

    with suggestion_columns[i % 2]:

        if st.button(
            suggestion,
            use_container_width=True
        ):

            st.session_state.pending_question = suggestion
            st.rerun()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("status"):

                st.caption(
                    f"Response type: {message['status']}"
                )


# --------------------------------------------------
# USER CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask anything about your selected event..."
)

if "pending_question" in st.session_state:

    question = st.session_state.pop("pending_question")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    answer, status = st.session_state.assistant.respond(
        selected_event,
        question
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "status": status
    })

    # Save every conversation
    save_log(
        selected_event,
        question,
        answer
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.caption(
            f"Response type: {status}"
        )

        if status == "Escalated":

            st.warning(
                "Human support escalation required."
            )

    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EventAssist AI | RAG-based retrieval | "
    "Event-level data isolation | Mock booking API"
)
