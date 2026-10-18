from fastapi.testclient import TestClient
from aureon.api import app

c = TestClient(app)


def test_roundtrip():
    r = c.post("/api/ask", json={"query": "q", "context": {"slice": "arithmetic", "truth": "x"}}).json()
    assert r["action"] in ("answer", "abstain") and "id" in r
    assert c.post("/api/feedback", json={"decision_id": r["id"], "correct": True}).json()["ok"]
    assert "weights" in c.get("/api/status").json()


def test_cors_not_wildcard():
    h = c.options("/api/ask", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}).headers
    assert h.get("access-control-allow-origin") != "*" and "evil" not in h.get("access-control-allow-origin", "")


def test_gaps_endpoint():
    assert isinstance(c.get("/api/gaps").json(), list)
