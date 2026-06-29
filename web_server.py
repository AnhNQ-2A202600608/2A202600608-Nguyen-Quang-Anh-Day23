import os
import json
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

# Load env vars safely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import core lab systems
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Scenario, Route, initial_state
from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.report import write_report
from langgraph_agent_lab.metrics import MetricsReport

PORT = 8080

def get_checkpoints_count():
    db_path = os.getenv("SQLITE_DB_PATH", "outputs/checkpoints.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    return 0

def run_single_ticket(query, route_expected, requires_approval, max_attempts=3, mock_approval_val=True):
    checkpointer = build_checkpointer("sqlite", os.getenv("SQLITE_DB_PATH", "outputs/checkpoints.db"))
    graph = build_graph(checkpointer=checkpointer)
    
    scenario = Scenario(
        id="html-ui-run",
        query=query,
        expected_route=Route(route_expected),
        requires_approval=requires_approval,
        max_attempts=max_attempts
    )
    state = initial_state(scenario)
    state["approval"] = {
        "approved": mock_approval_val,
        "reviewer": "HTML-Dashboard-User",
        "comment": "Approved via Web HTML Dashboard UI"
    }
    
    run_config = {"configurable": {"thread_id": f"thread-html-{route_expected}-{hash(query) % 10000}"}}
    final_state = graph.invoke(state, config=run_config)
    return final_state

class DashboardAPIHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Silence standard request logging for a cleaner console output
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        
        # Routing static assets or api
        if url.path == "/" or url.path == "/index.html":
            self.serve_static("web/index.html", "text/html")
        elif url.path == "/api/status":
            llm_provider = os.getenv("LLM_PROVIDER", "gemini")
            gemini_key_exists = bool(os.getenv("GEMINI_API_KEY"))
            checkpointer_kind = os.getenv("CHECKPOINTER", "sqlite")
            langsmith_active = os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
            
            db_checkpoints = get_checkpoints_count()
            
            self.send_json({
                "llm_provider": llm_provider,
                "gemini_key_exists": gemini_key_exists,
                "checkpointer_kind": checkpointer_kind,
                "langsmith_active": langsmith_active,
                "db_checkpoints": db_checkpoints,
                "sqlite_path": os.getenv("SQLITE_DB_PATH", "outputs/checkpoints.db")
            })
            
        elif url.path == "/api/artifacts":
            # Check file sizes and modification times
            files = ["outputs/metrics.json", "outputs/metrics_extended.json", "outputs/metrics_golden.json", "reports/lab_report.md", "reports/lab_report_extended.md", "reports/lab_report_golden.md", "outputs/checkpoints.db"]
            res = {}
            for f in files:
                p = Path(f)
                if p.exists():
                    size = p.stat().st_size
                    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    res[f] = {"exists": True, "size": f"{size/1024:.1f} KB" if size >= 1024 else f"{size} B", "modified": mtime}
                else:
                    res[f] = {"exists": False, "size": "N/A", "modified": "N/A"}
            self.send_json(res)
            
        elif url.path == "/api/load-metrics":
            query_params = parse_qs(url.query)
            spec = query_params.get("spec", ["scenarios"])[0]
            if "golden" in spec:
                metrics_file = "outputs/metrics_golden.json"
            elif "extended" in spec:
                metrics_file = "outputs/metrics_extended.json"
            else:
                metrics_file = "outputs/metrics.json"
            
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({"error": "Metrics file not found"}, 404)
        elif url.path == "/api/load-presets":
            query_params = parse_qs(url.query)
            spec = query_params.get("spec", ["scenarios"])[0]
            if "golden" in spec:
                spec_file = "data/sample/scenarios_golden.jsonl"
            elif "extended" in spec:
                spec_file = "data/sample/scenarios_extended.jsonl"
            else:
                spec_file = "data/sample/scenarios.jsonl"
                
            if os.path.exists(spec_file):
                scenarios = load_scenarios(spec_file)
                presets = {}
                for sc in scenarios:
                    presets[f"{sc.id}: {sc.query[:40]}..."] = {
                        "query": sc.query,
                        "route": sc.expected_route.value,
                        "requires_approval": sc.requires_approval,
                        "max_attempts": sc.max_attempts
                    }
                presets["Custom Ticket"] = {
                    "query": "My order has been delayed and I see a system connection error.",
                    "route": "error",
                    "requires_approval": false,
                    "max_attempts": 3
                }
                self.send_json(presets)
            else:
                self.send_json({"error": "Spec file not found"}, 404)
        elif url.path == "/api/last-state":
            path = Path("outputs/last_run_state.json")
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({"error": "No last state found"}, 404)
                
        elif url.path == "/api/load-file":
            query_params = parse_qs(url.query)
            filepath = query_params.get("path", [""])[0]
            if filepath in ["reports/lab_report.md", "reports/lab_report_extended.md", "reports/lab_report_golden.md", "configs/lab.yaml", "configs/lab_extended.yaml", "configs/lab_golden.yaml"]:
                p = Path(filepath)
                if p.exists():
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with open(p, "r", encoding="utf-8") as f:
                        self.wfile.write(f.read().encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(400)
                self.end_headers()
        elif url.path == "/api/download-file":
            query_params = parse_qs(url.query)
            filepath = query_params.get("path", [""])[0]
            if filepath in ["outputs/metrics.json", "outputs/metrics_extended.json", "outputs/metrics_golden.json", "reports/lab_report.md", "reports/lab_report_extended.md", "reports/lab_report_golden.md", "outputs/checkpoints.db"]:
                p = Path(filepath)
                if p.exists():
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Disposition', f'attachment; filename="{p.name}"')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with open(p, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        body = json.loads(post_data.decode('utf-8'))
        
        if url.path == "/api/run-ticket":
            query = body.get("query", "")
            route_expected = body.get("route_expected", "simple")
            requires_approval = bool(body.get("requires_approval", False))
            max_attempts = int(body.get("max_attempts", 3))
            mock_approval_val = bool(body.get("mock_approval_val", True))
            
            try:
                start_time = time.time()
                state = run_single_ticket(query, route_expected, requires_approval, max_attempts, mock_approval_val)
                latency = int((time.time() - start_time) * 1000)
                
                metric = metric_from_state(state, route_expected, requires_approval)
                
                payload = {
                    "state": dict(state),
                    "success": metric.success,
                    "retry_count": metric.retry_count,
                    "interrupt_count": metric.interrupt_count,
                    "latency_ms": latency
                }
                
                # Save as last state file
                Path("outputs").mkdir(exist_ok=True)
                with open("outputs/last_run_state.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                
                self.send_json(payload)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
                
        elif url.path == "/api/run-batch":
            spec = body.get("spec", "scenarios")
            if "golden" in spec:
                spec_file = "data/sample/scenarios_golden.jsonl"
                metrics_file = "outputs/metrics_golden.json"
            elif "extended" in spec:
                spec_file = "data/sample/scenarios_extended.jsonl"
                metrics_file = "outputs/metrics_extended.json"
            else:
                spec_file = "data/sample/scenarios.jsonl"
                metrics_file = "outputs/metrics.json"
            
            if os.path.exists(spec_file):
                try:
                    scenarios = load_scenarios(spec_file)
                    metrics = []
                    for sc in scenarios:
                        res = run_single_ticket(
                            query=sc.query,
                            route_expected=sc.expected_route.value,
                            requires_approval=sc.requires_approval,
                            max_attempts=sc.max_attempts,
                            mock_approval_val=True
                        )
                        metric = metric_from_state(res, sc.expected_route.value, sc.requires_approval)
                        metrics.append(metric)
                        
                    report = summarize_metrics(metrics)
                    Path("outputs").mkdir(exist_ok=True)
                    with open(metrics_file, "w", encoding="utf-8") as f:
                        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
                        
                    self.send_json({
                        "message": "Batch ran successfully",
                        "metrics": report.model_dump()
                    })
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"error": "Spec file not found"}, 404)
                
        elif url.path == "/api/generate-report":
            spec = body.get("spec", "scenarios")
            if "golden" in spec:
                metrics_file = "outputs/metrics_golden.json"
                report_file = "reports/lab_report_golden.md"
            elif "extended" in spec:
                metrics_file = "outputs/metrics_extended.json"
                report_file = "reports/lab_report_extended.md"
            else:
                metrics_file = "outputs/metrics.json"
                report_file = "reports/lab_report.md"
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    report_obj = MetricsReport.model_validate(data)
                    write_report(report_obj, report_file)
                    self.send_json({"message": "Report generated successfully", "path": report_file})
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"error": "Metrics file not found. Please run batch first."}, 400)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_static(self, filepath, content_type):
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(filepath, "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, DashboardAPIHandler)
    print("\n========================================================")
    print("LangGraph Support Agent Web Dashboard Server is running!")
    print(f"Local URL: http://localhost:{PORT}")
    print("========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    main()
