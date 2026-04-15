import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock

from app.interfaces.dev_api import app


# A fake graph that resolves instantly, so tests don't hit real LLMs
FAKE_OUTPUT = {
    "messages": [],
    "locked_route": "faq",
    "estimated_route": "faq",
    "confidence": 0.95,
    "routing_attempts": 1,
    "solve_attempts": 1,
    "max_solve_attempts": 3,
    "escalated_to_human": False,
    "retrieved": [],
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.anyio
async def test_chat_returns_202_with_job_id(client):
    with patch("app.interfaces.dev_api.graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_OUTPUT)

        r = await client.post("/chat", json={"text": "hello", "thread_id": "t1"})
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "pending"


@pytest.mark.anyio
async def test_poll_returns_done(client):
    with patch("app.interfaces.dev_api.graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_OUTPUT)

        post = await client.post("/chat", json={"text": "hello", "thread_id": "t1"})
        job_id = post.json()["job_id"]

        # Give the background task a moment to complete
        await asyncio.sleep(0.2)

        r = await client.get(f"/result/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "done"
        assert "answer" in r.json()


@pytest.mark.anyio
async def test_missing_text_returns_400(client):
    r = await client.post("/chat", json={"thread_id": "t1"})
    assert r.status_code == 400


@pytest.mark.anyio
async def test_auth_rejected_when_key_set(client, monkeypatch):
    monkeypatch.setattr("app.interfaces.dev_api.TEST_KEY", "secret")

    r = await client.post("/chat", json={"text": "hi", "thread_id": "t1"})
    assert r.status_code == 401

    r = await client.post(
        "/chat",
        json={"text": "hi", "thread_id": "t1"},
        headers={"X-Test-Key": "wrong"},
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_auth_accepted_with_correct_key(client, monkeypatch):
    monkeypatch.setattr("app.interfaces.dev_api.TEST_KEY", "secret")

    with patch("app.interfaces.dev_api.graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_OUTPUT)

        r = await client.post(
            "/chat",
            json={"text": "hi", "thread_id": "t1"},
            headers={"X-Test-Key": "secret"},
        )
        assert r.status_code == 202


@pytest.mark.anyio
async def test_result_404_for_unknown_job(client):
    r = await client.get("/result/nonexistent-job-id")
    assert r.status_code == 404
