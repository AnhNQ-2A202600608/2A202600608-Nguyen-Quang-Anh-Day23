"""
Golden Test Suite Grader
Runs 10 golden questions through the LangGraph support agent.
All Vietnamese output is written to JSON file, only ASCII is printed to console.
"""
import json
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer

GOLDEN_TESTS = [
    {
        "id": "gq_d10_01",
        "question": "Theo chính sách hoàn tiền hiện hành, khách hàng có tối đa bao nhiêu ngày làm việc để gửi yêu cầu hoàn tiền sau khi đơn được xác nhận?",
        "must_contain_any": ["7 ngày", "7 ngày làm việc"],
        "must_not_contain": ["14 ngày", "14 ngày làm việc"],
        "expect_top1_doc_id": "policy_refund_v4",
    },
    {
        "id": "gq_d10_02",
        "question": "Đâu là loại sản phẩm bị loại khỏi điều kiện hoàn tiền?",
        "must_contain_any": ["hàng kỹ thuật số", "license key", "subscription"],
        "must_not_contain": [],
        "expect_top1_doc_id": "policy_refund_v4",
    },
    {
        "id": "gq_d10_03",
        "question": "Finance Team xử lý yêu cầu hoàn tiền trong bao lâu?",
        "must_contain_any": ["3-5 ngày làm việc", "3 đến 5 ngày", "3 ngày", "5 ngày"],
        "must_not_contain": [],
        "expect_top1_doc_id": "policy_refund_v4",
    },
    {
        "id": "gq_d10_04",
        "question": "SLA phản hồi ban đầu cho ticket P1 là bao lâu?",
        "must_contain_any": ["15 phút", "15p", "15 minutes"],
        "must_not_contain": [],
        "expect_top1_doc_id": "sla_p1_2026",
    },
    {
        "id": "gq_d10_05",
        "question": "SLA resolution cho ticket P1 là bao nhiêu giờ?",
        "must_contain_any": ["4 giờ", "4h", "4 hours"],
        "must_not_contain": [],
        "expect_top1_doc_id": "sla_p1_2026",
    },
    {
        "id": "gq_d10_06",
        "question": "Nếu không có phản hồi với ticket P1 sau bao lâu thì hệ thống auto escalate?",
        "must_contain_any": ["10 phút", "10 minutes"],
        "must_not_contain": [],
        "expect_top1_doc_id": "sla_p1_2026",
    },
    {
        "id": "gq_d10_07",
        "question": "Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai liên tiếp?",
        "must_contain_any": ["5 lần", "5 times", "5 attempts"],
        "must_not_contain": [],
        "expect_top1_doc_id": "it_helpdesk_faq",
    },
    {
        "id": "gq_d10_08",
        "question": "VPN cho phép kết nối tối đa bao nhiêu thiết bị cùng lúc?",
        "must_contain_any": ["2 thiết bị", "2 device", "2 devices", "hai thiết bị"],
        "must_not_contain": [],
        "expect_top1_doc_id": "it_helpdesk_faq",
    },
    {
        "id": "gq_d10_09",
        "question": "Nhân viên dưới 3 năm kinh nghiệm được bao nhiêu ngày phép năm theo chính sách HR 2026?",
        "must_contain_any": ["12 ngày", "12 ngày phép năm", "12 days"],
        "must_not_contain": ["10 ngày phép năm", "10 ngày phép"],
        "expect_top1_doc_id": "hr_leave_policy",
    },
    {
        "id": "gq_d10_10",
        "question": "Level 4 Admin Access yêu cầu phê duyệt bởi ai?",
        "must_contain_any": ["IT Manager", "CISO"],
        "must_not_contain": [],
        "expect_top1_doc_id": "access_control_sop",
    },
]


def run_single_question(graph, question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "thread_id": thread_id,
        "scenario_id": thread_id,
        "query": question,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": 3,
        "final_answer": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }
    result = graph.invoke(initial, config=config)
    return result


def check_answer(answer: str, test: dict) -> tuple:
    failures = []
    answer_lower = answer.lower()

    if test["must_contain_any"]:
        if not any(kw.lower() in answer_lower for kw in test["must_contain_any"]):
            failures.append(f"must_contain_any: expected one of {test['must_contain_any']}")

    for bad in test.get("must_not_contain", []):
        if bad.lower() in answer_lower:
            failures.append(f"must_not_contain: found forbidden '{bad}'")

    return len(failures) == 0, failures


def grade():
    checkpointer = build_checkpointer(kind="sqlite", database_url="outputs/checkpoints.db")
    graph = build_graph(checkpointer=checkpointer)

    results = []
    passed = 0
    failed = 0

    print("=" * 60)
    print("  GOLDEN TEST SUITE - LangGraph Support Agent")
    print("=" * 60)

    for i, test in enumerate(GOLDEN_TESTS, 1):
        thread_id = f"golden-{test['id']}"
        print(f"[{i:02d}/{len(GOLDEN_TESTS)}] {test['id']} | doc: {test['expect_top1_doc_id']}", flush=True)

        try:
            state = run_single_question(graph, test["question"], thread_id)
        except Exception as e:
            err = str(e)
            print(f"  ERROR: {err[:80]}")
            failed += 1
            results.append({
                "id": test["id"],
                "status": "ERROR",
                "route": "?",
                "answer": "",
                "doc_retrieved": "",
                "failures": [err],
            })
            continue

        answer = state.get("final_answer") or ""
        tool_results = state.get("tool_results", [])
        doc_retrieved = ""
        if tool_results:
            last = str(tool_results[-1])
            m = re.search(r"doc_id[=:\s]+(\w+)", last, re.IGNORECASE)
            if m:
                doc_retrieved = m.group(1)

        ok, failures = check_answer(answer, test)
        doc_ok = (doc_retrieved == test["expect_top1_doc_id"]) if doc_retrieved else None

        status = "PASS" if ok else "FAIL"
        route = state.get("route", "?")

        if ok:
            passed += 1
            print(f"  PASS | route={route} | doc={doc_retrieved or '-'}")
        else:
            failed += 1
            print(f"  FAIL | route={route} | doc={doc_retrieved or '-'}")
            for f_msg in failures:
                # encode to ASCII-safe for Windows console
                safe_msg = f_msg.encode("ascii", errors="replace").decode("ascii")
                print(f"       {safe_msg}")

        results.append({
            "id": test["id"],
            "question": test["question"],
            "status": status,
            "route": route,
            "answer": answer,
            "doc_retrieved": doc_retrieved,
            "expected_doc": test["expect_top1_doc_id"],
            "doc_match": doc_ok,
            "failures": failures,
        })

    print("=" * 60)
    score_pct = passed / len(GOLDEN_TESTS) * 100
    print(f"  RESULTS: {passed}/{len(GOLDEN_TESTS)} PASSED  |  SCORE: {score_pct:.1f}%")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/golden_grading_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(GOLDEN_TESTS),
            "passed": passed,
            "failed": failed,
            "score_pct": score_pct,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {out_path}")
    return passed == len(GOLDEN_TESTS)


if __name__ == "__main__":
    success = grade()
    sys.exit(0 if success else 1)
