"""vLLM nhận beam width trong một request; API giữ deadline chung và log kết quả."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from submission.app import create_app
from submission.backend import ModelOutputError, VLLMBackend
from submission.config import ServingConfig
from submission.schema import Query
from tests.test_consistency import candidate


def choices(answers):
    return [{"index": i, "message": {"content": candidate(answer)["raw_output"]},
             "finish_reason": "stop"} for i, answer in enumerate(answers)]


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_samples_one_call_and_return_winning_explanation(self):
        calls = []
        outputs = choices(["8", "6", "6.0", "3*2", "9"])
        outputs[1]["message"]["content"] = candidate("6", explanation="F=ma=2*3=6 N.")["raw_output"]
        def handle(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": list(reversed(outputs)),
                                             "usage": {"completion_tokens": 100}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://test") as client:
            backend = VLLMBackend(ServingConfig(beam_width=1, self_consistency_samples=5), client)
            result, info = await backend.predict(Query(query_id="q", query="Tính lực."))
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["n"], calls[0]["temperature"], calls[0]["top_p"]), (5, 0.7, 0.8))
        self.assertEqual(result["answer"], "6")
        self.assertEqual(result["explanation"], "F=ma=2*3=6 N.")
        self.assertEqual(result["query_id"], "q")
        self.assertEqual(info["self_consistency"]["vote_count"], 3)
        self.assertEqual(info["usage"]["completion_tokens"], 100)

    async def test_all_unfinished_is_an_output_error_with_candidate_logs(self):
        outputs = choices(["1"] * 5)
        for output in outputs:
            output["finish_reason"] = "length"
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"choices": outputs})), base_url="http://test") as client:
            with self.assertRaises(ModelOutputError) as raised:
                await VLLMBackend(ServingConfig(beam_width=1, self_consistency_samples=5), client).predict(
                    Query(query_id="q", query="Câu hỏi"))
        self.assertEqual(raised.exception.details["self_consistency"]["valid_samples"], 0)
        self.assertEqual(len(raised.exception.details["self_consistency"]["candidates"]), 5)

    async def test_missing_or_duplicate_choices_are_rejected(self):
        for outputs in (choices(["1"]), [choices(["1"])[0]] * 5):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"choices": outputs})), base_url="http://test") as client:
                with self.subTest(count=len(outputs)), self.assertRaises(ModelOutputError):
                    await VLLMBackend(ServingConfig(beam_width=1, self_consistency_samples=5), client).predict(Query(query_id="q", query="Câu hỏi"))

    async def test_one_sample_keeps_greedy_request(self):
        sent = []
        def handle(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": choices(["6"])})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://test") as client:
            result, _ = await VLLMBackend(ServingConfig(beam_width=1, self_consistency_samples=1), client).predict(
                Query(query_id="q", query="Câu hỏi"))
        self.assertEqual((sent[0]["n"], sent[0]["temperature"], sent[0]["top_p"]), (1, 0.0, 1.0))
        self.assertEqual(result["answer"], "6")

    async def test_beam_search_sends_width_and_selects_first_ranked_choice(self):
        sent = []
        outputs = choices(["1", "2", "3"])
        def handle(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": list(reversed(outputs))})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://test") as client:
            result, info = await VLLMBackend(ServingConfig(beam_width=3), client).predict(
                Query(query_id="q", query="Câu hỏi"))
        self.assertEqual((sent[0]["use_beam_search"], sent[0]["n"], sent[0]["temperature"]), (True, 3, 0.0))
        self.assertEqual(result["answer"], "1")
        self.assertEqual(info["beam_search"]["selected_index"], 0)


class APITests(unittest.TestCase):
    def test_total_request_deadline_is_not_multiplied_by_sample_count(self):
        class SlowBackend:
            calls = 0
            async def models(self): return {"object": "list", "data": []}
            async def close(self): pass
            async def predict(self, query):
                self.calls += 1
                await asyncio.sleep(0.2)
        with tempfile.TemporaryDirectory() as directory:
            cfg = ServingConfig(log_dir=directory, request_deadline_seconds=0.02, beam_width=3)
            engine = SlowBackend()
            with TestClient(create_app(cfg, backend=engine)) as client:
                self.assertEqual(client.get("/healthz").json()["decoding"]["beam_width"], 3)
                response = client.post("/predict", json={"query_id": "q", "query": "Câu hỏi"})
            self.assertEqual(response.status_code, 504)
            self.assertEqual(engine.calls, 1)
            event = json.loads((Path(directory) / "requests.jsonl").read_text())
            self.assertEqual(event["status"], "timeout")

    def test_invalid_candidates_are_logged_without_returning_them_to_user(self):
        class InvalidBackend:
            async def models(self): return {"object": "list", "data": []}
            async def close(self): pass
            async def predict(self, query):
                raise ModelOutputError("no valid samples", {"self_consistency": {"valid_samples": 0}})
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(ServingConfig(log_dir=directory, beam_width=3), backend=InvalidBackend())) as client:
                response = client.post("/predict", json={"query_id": "q", "query": "Câu hỏi"})
            self.assertEqual(response.status_code, 502)
            event = json.loads((Path(directory) / "requests.jsonl").read_text())
            self.assertEqual(event["self_consistency"]["valid_samples"], 0)
