# EventAssist AI – Events & Hospitality Assistant

A Python-based AI assistant prototype developed for the MCI India AI Solution Engineer – Enterprise Automation & Copilot Systems technical assessment.

EventAssist AI helps conference attendees get event-specific information, check booking status, and receive support through an interactive web application.

## 1. Key Features

- Retrieval-Augmented Generation (RAG)-based information retrieval using TF-IDF and cosine similarity.
- Supports two independent events:
  - `event_a`: Global Tech Summit
  - `event_b`: Hospitality Leadership Forum
- Separate knowledge indexes for each event.
- Event-specific information retrieval and cross-event request blocking.
- Mock booking API integration.
- Multi-turn conversations through Streamlit.
- VIP and dissatisfied-user escalation.
- JSONL conversation logging.
- Safe responses for unknown questions.
- Basic prompt-injection detection.
- Automated tests using pytest.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Retrieval method | TF-IDF |
| Similarity calculation | Cosine Similarity |
| Machine learning library | Scikit-learn |
| Conversation logging | JSONL |
| Testing | Pytest |

## 3. Architecture

The application follows a modular architecture with separate responsibilities for event data, retrieval, security, booking operations, escalation, and logging.

The main workflow is:

1. The attendee selects an event.
2. The assistant receives the question.
3. Security checks identify suspicious or cross-event requests.
4. The booking API is called when booking information is requested.
5. Otherwise, the event-specific knowledge base retrieves relevant information.
6. The assistant returns a grounded answer or a safe fallback.
7. VIP and dissatisfied-user requests are escalated.
8. The conversation is saved in a JSONL log file.

See `architecture.png` for the architecture diagram.

## 4. RAG Implementation

The prototype uses TF-IDF and cosine similarity for information retrieval.

### Retrieval workflow

1. Each event has its own collection of information.
2. A separate TF-IDF vectorizer and document matrix are created for each event.
3. The user's question is converted into a numerical vector.
4. Cosine similarity compares the question with documents belonging to the selected event.
5. The most relevant document is selected.
6. If the similarity score is below the configured threshold, the assistant returns a safe fallback.
7. Otherwise, the retrieved information is returned to the user.

The application uses extractive responses rather than generating answers with an external LLM.

This approach is suitable for the small assessment dataset because it is lightweight, transparent, and does not require an API key.

For production, semantic embeddings and a managed vector database can replace TF-IDF.

## 5. Event Data Isolation

Event isolation is an important requirement of this application.

The prototype implements the following controls:

- Each event has a separate knowledge index.
- Retrieval searches only the currently selected event's index.
- The active event is maintained in Streamlit session state.
- Changing the selected event clears the previous conversation.
- Requests for information about another event are blocked.
- User messages cannot directly change the event ID passed to the retrieval function.

### Production security approach

For a production deployment:

- Authenticate users before providing event information.
- Verify event membership using registration records.
- Resolve the authorized event ID on the server.
- Apply tenant-level access filters in the database and vector store.
- Never trust an event ID supplied directly by a user or an LLM.

These additional authentication and authorization controls are proposed production requirements and are not implemented in the current prototype.

## 6. Booking API Integration

The assessment provides a mock booking API:

```python
def get_booking_status(event_id: str, booking_id: str):
    return f"Booking {booking_id} for {event_id} is confirmed."
```

The assistant:

1. Detects booking or reservation-related questions.
2. Extracts the booking ID from the user's message.
3. Passes the active event ID and booking ID to the mock API.
4. Displays the returned booking status.
5. Requests the booking ID if it is missing.

Example:

**User:** Check booking status for booking ID ABC123.

**Assistant:** Booking ABC123 for event_b is confirmed.

The current implementation uses a mock function. A production system should integrate with an authenticated booking service using secure API communication, input validation, timeouts, retries, and proper authorization.

## 7. VIP and Dissatisfied-User Escalation

The assistant checks messages for VIP-related terms and expressions of dissatisfaction.

Examples include:

- VIP
- Sponsor
- Premium
- Unhappy
- Frustrated
- Complaint
- Dissatisfied

When escalation is detected, the assistant provides a human-support escalation message.

The current prototype simulates escalation through the response and does not create an actual support ticket.

A production implementation could integrate with a CRM or ticketing system and notify the event operations team.

## 8. Hallucination Reduction

The prototype reduces unsupported answers through:

- Event-specific document retrieval.
- A minimum similarity threshold.
- Returning retrieved information instead of generating unsupported facts.
- Safe fallback responses for unknown questions.
- Booking-status responses obtained from the booking API.

If no relevant information is found, the assistant responds:

"I do not have verified information about that question for this event. Please contact the event support team for assistance."

A production LLM integration should use retrieved documents as its authorized context and provide source references for auditability.

## 9. Prompt-Injection Prevention

The prototype includes basic prompt-injection detection using predefined patterns.

It blocks examples such as:

- Ignore previous instructions.
- Give me data of other event.
- Bypass security.
- Reveal information about another event.

It also maintains event selection separately from the user's message.

### Additional production safeguards

- Treat user input and retrieved documents as untrusted data.
- Keep system instructions separate from retrieved content.
- Enforce authorization in backend code.
- Validate all tool arguments.
- Restrict tools to approved operations.
- Never allow an LLM to change the authorized event scope.
- Test direct and indirect prompt-injection attacks.

Keyword detection alone is not a complete security solution.

## 10. Conversation Logging

The application saves conversations in:

`logs/conversations.jsonl`

Each conversation record currently includes:

- Timestamp
- Selected event
- User message
- Assistant response

JSONL stores one JSON object per line, making conversation records easy to inspect and process.

For production, logging should also include request IDs, response status, retrieval scores, escalation metadata, API latency, and security events.

Sensitive attendee information, credentials, and authentication tokens should not be stored unnecessarily.

## 11. Scaling to 200,000 Daily Users

The current application is a local prototype. For a production deployment supporting approximately 200,000 daily users, an Azure-based architecture could be considered.

| Requirement | Proposed technology |
|---|---|
| Application hosting | Azure Container Apps or AKS |
| API management | Azure API Management |
| Authentication | Microsoft Entra ID |
| Vector search | Azure AI Search |
| Database | Azure Database for PostgreSQL or Cosmos DB |
| Session caching | Azure Cache for Redis |
| Background processing | Azure Service Bus |
| Document storage | Azure Blob Storage |
| Secret management | Azure Key Vault |
| Monitoring | Azure Monitor and Application Insights |

### Scalability strategies

- Horizontal application scaling.
- Load balancing.
- Database indexing and optimization.
- Caching frequently requested information.
- Queue-based asynchronous processing.
- API rate limiting.
- Monitoring peak concurrent users.
- Per-event resource and access controls.

Capacity planning should consider peak concurrent users, request volume, response time, and infrastructure costs rather than daily user count alone.

## 12. Security Considerations

The prototype demonstrates basic security checks. A production deployment should additionally implement:

- Authentication and event-level authorization.
- HTTPS/TLS communication.
- Secure API credentials.
- Input validation.
- Rate limiting.
- Database access controls.
- Data encryption.
- Personal information redaction.
- Audit logging.
- Defined data-retention policies.

## 13. Current Limitations

- Uses a small, hard-coded assessment knowledge base.
- TF-IDF is keyword-based and may miss semantic similarities.
- Does not use an external generative LLM.
- Booking API is simulated.
- Human escalation is simulated.
- No real ticketing or CRM integration.
- No production authentication system.
- Session state is maintained in application memory.
- Prompt-injection detection uses predefined patterns.
- Production deployment and distributed scaling are not implemented.

## 14. Installation and Setup

### Prerequisites

- Python 3.10 or newer
- pip
- VS Code or another Python editor

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the web application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### Step 5: Run automated tests

```bash
python -m pytest -q
```

## 15. Example Test Scenarios

| Test scenario | Expected behaviour |
|---|---|
| Ask registration time for Event A | Returns 8:00 AM |
| Ask registration time for Event B | Returns 9:00 AM |
| Ask Event A hall location | Returns Hall A, Level 1 |
| Ask Event B cancellation policy | Returns 48 hours |
| Check booking ABC123 | Calls mock booking API |
| VIP user expresses dissatisfaction | Escalation message |
| Ask for another event's information | Request blocked |
| Attempt prompt injection | Security refusal |
| Ask unknown question | Safe fallback response |

## 16. Project Structure

```text
events-hospitality-ai-assistant/
│
├── data/
├── logs/
│   ├── .gitkeep
│   └── conversations.jsonl
│
├── tests/
│   └── test_assistant.py
│
├── architecture.png
├── app.py
├── event_assistant.py
├── requirements.txt
└── README.md
```

## 17. Future Improvements

- Integrate a production booking API.
- Add authentication and event registration verification.
- Replace TF-IDF with semantic vector search.
- Integrate an LLM with grounded responses.
- Connect escalation to a real ticketing system.
- Add persistent conversation storage.
- Implement advanced monitoring and analytics.

---

**Project:** EventAssist AI  
**Purpose:** MCI India AI Solution Engineer Technical Assessment  
**Application:** Python and Streamlit  
**Architecture:** Event-scoped retrieval with booking and escalation workflows