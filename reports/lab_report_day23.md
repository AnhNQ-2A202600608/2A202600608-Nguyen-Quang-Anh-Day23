# BÁO CÁO BÀI LAB DAY 23 — LangGraph Agentic Orchestration with RAG & Human-in-the-Loop

**Học viên:** Nguyễn Quang Anh  
**Mã học viên:** 2A202600608  
**Ngày nộp:** 29/06/2026  
**Môn học:** Advanced AI Engineering — VinAI Action Program  
**Điểm kiểm thử golden set:** **10/10 PASS (100%)**

---

## 1. Tổng Quan Bài Lab

Bài lab yêu cầu xây dựng một **LangGraph Agent** hoàn chỉnh mô phỏng hệ thống hỗ trợ khách hàng (Support Agent) với các tính năng:

- Phân loại yêu cầu (intent classification) bằng LLM có structured output
- Định tuyến đa luồng (multi-route workflow): `simple`, `tool`, `missing_info`, `risky`, `error`
- Tích hợp vòng lặp retry với dead-letter queue
- Human-in-the-Loop (HITL) thông qua `interrupt()` của LangGraph
- Ghi checkpoint SQLite để traceability
- Đánh giá bằng bộ golden test 10 câu hỏi

---

## 2. Kiến Trúc Đồ Thị LangGraph

```
START
  └─► intake_node          # Chuẩn hóa query đầu vào
        └─► classify_node  # LLM phân loại route (structured output)
              ├─[simple]──► answer_node ──► finalize_node ──► END
              │
              ├─[tool]────► tool_node ──► evaluate_node
              │               ├─[success]──► answer_node ──► finalize_node ──► END
              │               └─[needs_retry]─► retry_node
              │                                   ├─[retry]──► tool_node (vòng lặp)
              │                                   └─[dead_letter]─► dead_letter_node ──► finalize_node ──► END
              │
              ├─[missing_info]─► clarify_node ──► finalize_node ──► END
              │
              ├─[risky]───► risky_action_node ──► approval_node
              │               ├─[approved]──► tool_node ──► evaluate_node ──► ...
              │               └─[rejected]──► clarify_node ──► finalize_node ──► END
              │
              └─[error]───► retry_node ──► ...
```

**Tổng cộng:** 10 nodes, 4 conditional edges, 1 entry point, 1 terminal node.

---

## 3. Chi Tiết Cài Đặt

### 3.1. State Schema (`state.py`)

```python
class AgentState(TypedDict):
    thread_id: str
    scenario_id: str
    query: str
    route: str               # simple | tool | missing_info | risky | error
    risk_level: str          # low | high
    attempt: int             # số lần retry hiện tại
    max_attempts: int        # ngưỡng retry tối đa
    final_answer: Optional[str]
    messages: list[str]
    tool_results: list[str]
    errors: list[str]
    events: list[dict]       # audit trail đầy đủ
    # HITL fields
    proposed_action: Optional[str]
    approval: Optional[dict]
    evaluation_result: Optional[str]
    pending_question: Optional[str]
```

Mỗi event được ghi nhận với timestamp, node name, event_type, và payload tùy ý.

### 3.2. classify_node — LLM Structured Output

```python
class ClassificationResult(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]

structured_llm = llm.with_structured_output(ClassificationResult)
result = structured_llm.invoke(classification_prompt)
```

- **Ưu tiên phân loại:** `risky > error > missing_info > tool > simple`
- **Fallback heuristic:** Khi LLM không khả dụng, dùng keyword-based classifier với từ khóa tiếng Việt và tiếng Anh
- **Vietnamese policy detection:** Nhận diện các query như "hoàn tiền", "chính sách", "SLA", "ngày làm việc", v.v. → route `tool`

### 3.3. tool_node — Knowledge Retrieval

`tool_node` đóng vai trò **document retriever** cho 5 knowledge bases:

| doc_id | Nội dung | Keywords kích hoạt |
|--------|----------|-------------------|
| `policy_refund_v4` | Chính sách hoàn tiền: 7 ngày làm việc, exclusions (digital goods, license key, subscription), Finance Team 3-5 ngày | hoàn tiền, refund, sản phẩm bị loại, finance |
| `sla_p1_2026` | SLA P1: phản hồi 15 phút, resolution 4 giờ, auto escalate sau 10 phút | sla, p1, escalate, resolution, phản hồi |
| `it_helpdesk_faq` | IT FAQ: tài khoản khóa sau 5 lần sai, VPN 2 thiết bị | khóa, đăng nhập, vpn, thiết bị |
| `hr_leave_policy` | HR 2026: nhân viên < 3 năm → 12 ngày phép năm | phép, nghỉ phép, kinh nghiệm |
| `access_control_sop` | Level 4 Admin Access: phê duyệt bởi IT Manager hoặc CISO | level 4, admin access, sop, cấp quyền |

Đối với route `error` với `attempt < 2`: node simulate **transient failure** để kích hoạt vòng lặp retry.

### 3.4. evaluate_node — LLM-as-Judge (Bonus)

```python
# Heuristic mode (default)
evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"

# LLM-as-Judge mode (USE_LLM_JUDGE=true)
response = llm.invoke(f"Evaluate: {latest_result}. Reply 'success' or 'needs_retry'")
```

### 3.5. approval_node — Human-in-the-Loop (HITL)

```python
# HITL mode (LANGGRAPH_INTERRUPT=true)
decision = interrupt({
    "question": "Approve this risky action?",
    "proposed_action": state.get("proposed_action", ""),
})
approval = {
    "approved": decision.get("approved", False),
    "reviewer": decision.get("reviewer", "human"),
    "comment": decision.get("comment", ""),
}

# Mock mode (offline/CI)
approval = {"approved": True, "reviewer": "mock-reviewer", "comment": "Auto-approved (mock)"}
```

Khi `LANGGRAPH_INTERRUPT=true`, workflow **dừng lại** tại `approval_node` và chờ input từ người dùng thực. Đây là HITL thực sự theo chuẩn LangGraph.

### 3.6. Retry Loop & Dead Letter

```
tool_node → evaluate_node
    ├─ success → answer_node
    └─ needs_retry → retry_node
           ├─ attempt < max_attempts → tool_node (retry)
           └─ attempt >= max_attempts → dead_letter_node
```

Sau `max_attempts` (mặc định 3), workflow chuyển sang `dead_letter_node` và ghi nhận escalation.

---

## 4. Persistence — SQLite Checkpoint

```python
# persistence.py
conn = sqlite3.connect("outputs/checkpoints.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")  # WAL mode cho concurrent safety
checkpointer = SqliteSaver(conn=conn)

# Sử dụng
graph = build_graph(checkpointer=checkpointer)
result = graph.invoke(state, config={"configurable": {"thread_id": "..."}})
```

Mỗi lần chạy được lưu vào SQLite với `thread_id` để có thể resume và trace lại.

---

## 5. Kết Quả Chạy — Metrics

### 5.1 Batch Run (Scenario Suites)

Hệ thống hỗ trợ chạy batch qua file JSONL:

```yaml
# configs/lab.yaml
scenarios_file: data/sample/scenarios.jsonl
checkpointer: sqlite
database_url: outputs/checkpoints.db
```

Kết quả được export ra `outputs/metrics.json`, `outputs/metrics_extended.json`, `outputs/metrics_golden.json` với:
- Latency trung bình mỗi route
- Tỷ lệ success/failure
- Số lần retry phân bổ theo scenario
- Node path trace cho từng run

### 5.2 Golden Test Set — 10/10 PASS (100%)

Kết quả kiểm thử với bộ 10 câu hỏi nghiệm thu:

| # | ID | Câu hỏi (rút gọn) | Route | Doc Retrieved | Status |
|---|----|--------------------|-------|---------------|--------|
| 01 | gq_d10_01 | Hoàn tiền tối đa bao nhiêu ngày? | tool | policy_refund_v4 | ✅ PASS |
| 02 | gq_d10_02 | Loại sản phẩm bị loại khỏi hoàn tiền? | tool | policy_refund_v4 | ✅ PASS |
| 03 | gq_d10_03 | Finance Team xử lý trong bao lâu? | tool | policy_refund_v4 | ✅ PASS |
| 04 | gq_d10_04 | SLA phản hồi P1 là bao lâu? | tool | sla_p1_2026 | ✅ PASS |
| 05 | gq_d10_05 | SLA resolution P1 là bao nhiêu giờ? | tool | sla_p1_2026 | ✅ PASS |
| 06 | gq_d10_06 | Auto escalate P1 sau bao lâu? | tool | sla_p1_2026 | ✅ PASS |
| 07 | gq_d10_07 | Tài khoản khóa sau bao nhiêu lần sai? | tool | it_helpdesk_faq | ✅ PASS |
| 08 | gq_d10_08 | VPN tối đa bao nhiêu thiết bị? | tool | it_helpdesk_faq | ✅ PASS |
| 09 | gq_d10_09 | Nhân viên < 3 năm được bao nhiêu ngày phép? | tool | hr_leave_policy | ✅ PASS |
| 10 | gq_d10_10 | Level 4 Admin Access phê duyệt bởi ai? | tool | access_control_sop | ✅ PASS |

**Tổng kết:** 10/10 câu hỏi PASS. Score: **100.0%**

Chi tiết câu trả lời mẫu cho **gq_d10_01**:
> *"Theo chính sách hoàn tiền (policy_refund_v4): Khách hàng có tối đa **7 ngày làm việc** sau khi đơn hàng được xác nhận để gửi yêu cầu hoàn tiền."*

---

## 6. Dashboard Web UI

Dự án tích hợp một **HTML/CSS/JS Dashboard** độc lập (không dùng Streamlit) phục vụ giám sát và vận hành:

### Tính năng dashboard:
1. **Overview Tab** — Sơ đồ kiến trúc LangGraph dạng Mermaid flowchart (render thực)
2. **Single Run Analyst** — Gửi câu hỏi trực tiếp, xem kết quả real-time (route, answer, latency)
3. **Scenario Batch Runs** — Upload/chọn file JSONL, chạy batch và xem metrics tổng hợp
4. **State & Trace** — Xem state JSON và event trace của từng thread_id
5. **Artifacts Preview** — Preview file metrics JSON, reports Markdown trong browser

### Kiến trúc client-server:
```
Browser (HTML/CSS/JS)
    ↕ REST API (JSON)
web_server.py (Python HTTP Server — port 8080)
    ↕ Python calls
LangGraph Agent (src/)
    ↕ SQLite WAL
outputs/checkpoints.db
```

**Khởi động:** `python web_server.py` → truy cập `http://127.0.0.1:8080/`

---

## 7. Cấu Trúc Dự Án

```
2A202600608-Nguyen-Quang-Anh-Day23/
├── src/langgraph_agent_lab/
│   ├── state.py          # AgentState TypedDict + make_event helper
│   ├── nodes.py          # 10 node functions (classify, tool, answer, approval, ...)
│   ├── routing.py        # 4 conditional edge functions
│   ├── graph.py          # build_graph() → CompiledGraph
│   ├── persistence.py    # build_checkpointer() → SQLite / Memory / None
│   ├── llm.py            # get_llm() → ChatOpenAI / ChatGoogleGenerativeAI
│   └── report.py         # generate_report() → Markdown report
├── configs/
│   ├── lab.yaml           # Base config (10 scenarios)
│   ├── lab_extended.yaml  # Extended config (15 scenarios)
│   └── lab_golden.yaml    # Golden test config (10 questions)
├── data/sample/
│   ├── scenarios.jsonl          # 10 base test scenarios
│   ├── scenarios_extended.jsonl # 15 extended scenarios
│   └── scenarios_golden.jsonl   # 10 golden evaluation questions
├── outputs/
│   ├── checkpoints.db           # SQLite checkpoint database
│   ├── metrics.json             # Base run metrics
│   ├── metrics_extended.json    # Extended run metrics
│   ├── metrics_golden.json      # Golden run metrics
│   ├── last_run_state.json      # Last single-run state (for dashboard)
│   └── golden_grading_results.json  # 10/10 PASS grading results
├── reports/
│   ├── lab_report.md            # Base run report
│   ├── lab_report_extended.md   # Extended run report
│   └── lab_report_golden.md     # Golden run report
├── web/
│   └── index.html               # HTML Dashboard (single-file SPA)
├── web_server.py                # HTTP server + REST API backend
├── grade_golden.py              # Golden test grader script
├── conftest.py                  # Pytest fixtures
└── tests/                       # Unit tests
```

---

## 8. Hướng Dẫn Chạy

### Cài đặt
```bash
pip install -e ".[dev]"
# hoặc
pip install langgraph langchain-openai langchain-google-genai langgraph-checkpoint-sqlite
```

### Cấu hình API Key
```bash
# .env
GEMINI_API_KEY=your_key_here
# hoặc
OPENAI_API_KEY=your_key_here
```

### Chạy batch scenarios
```bash
python -m langgraph_agent_lab.report --config configs/lab.yaml
```

### Chạy golden grader
```bash
python grade_golden.py
# Output: outputs/golden_grading_results.json
```

### Chạy Dashboard Web
```bash
python web_server.py
# Mở http://127.0.0.1:8080/
```

### Chạy unit tests
```bash
pytest tests/ -v
```

### Bật HITL thực (Human-in-the-Loop)
```bash
LANGGRAPH_INTERRUPT=true python -c "
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
graph = build_graph(checkpointer=build_checkpointer('sqlite'))
# Gửi risky query → graph sẽ dừng tại approval_node và chờ input
"
```

---

## 9. Kiểm Thử

### Unit Tests
```bash
pytest tests/ -v
```

Các test bao gồm:
- `test_state.py` — kiểm tra AgentState schema và make_event
- `test_nodes.py` — kiểm tra từng node function độc lập
- `test_graph.py` — kiểm tra build_graph và compilation
- `test_routing.py` — kiểm tra conditional edge functions

### Integration Tests (Golden Set)
```bash
python grade_golden.py
```

**Kết quả:** 10/10 PASS (100%)

---

## 10. Những Điểm Kỹ Thuật Nổi Bật

### 10.1 Vietnamese NLP Routing
Classify node nhận diện được queries tiếng Việt liên quan đến chính sách, không cần NLP library riêng mà dùng keyword matching kết hợp LLM:

```python
# Fallback classifier — Vietnamese keywords → tool route
tool_keywords_vi = [
    "chính sách", "hoàn tiền", "bao nhiêu", "bao lâu",
    "ngày làm việc", "điều kiện", "phản hồi", "vpn",
    "thiết bị", "ngày phép", "kinh nghiệm", "admin access",
    "phê duyệt", "finance team", "xử lý",
]
```

### 10.2 Conditional Retry Loop
Vòng lặp retry được implement qua conditional edge, không phải while loop:
```python
def route_after_retry(state: AgentState) -> str:
    if state.get("attempt", 0) >= state.get("max_attempts", 3):
        return "dead_letter"
    return "tool"
```

### 10.3 SQLite WAL Mode
```python
conn.execute("PRAGMA journal_mode=WAL")
```
WAL (Write-Ahead Logging) cho phép concurrent reads trong khi có write, quan trọng khi dashboard và agent chạy đồng thời.

### 10.4 LLM Provider Abstraction
```python
def get_llm():
    if os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini")
    raise RuntimeError("No LLM API key configured")
```

---

## 11. Tổng Kết

| Tiêu chí | Yêu cầu | Kết quả |
|----------|---------|---------|
| LangGraph workflow | 5 routes, multi-node | ✅ 10 nodes, 4 conditional edges |
| LLM integration | classify + answer | ✅ Gemini/OpenAI với structured output |
| Human-in-the-Loop | interrupt() tại approval | ✅ `LANGGRAPH_INTERRUPT=true` |
| SQLite checkpoint | WAL mode | ✅ `outputs/checkpoints.db` |
| Retry loop | max 3 attempts, dead letter | ✅ Implement đầy đủ |
| Golden test | 10/10 PASS | ✅ **100% score** |
| Dashboard UI | Web interface | ✅ HTML SPA tại port 8080 |
| Report generation | Markdown report | ✅ `reports/lab_report_golden.md` |

**Bài lab hoàn thành đầy đủ tất cả yêu cầu với điểm kiểm thử 10/10 (100%).**

---

*Nguyễn Quang Anh — 2A202600608 — VinAI Action Program Day 23*
