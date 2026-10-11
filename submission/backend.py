"""Gọi đúng vLLM self-hosted một lần; không retry, không dùng nhãn test."""

import os

import httpx

from baseline.data import prompt_messages
from .schema import competition_result


class VLLMBackend:
    def __init__(self, config, client=None):
        self.config = config
        key = os.environ.get("VIPRA_BACKEND_API_KEY")
        self.client = client or httpx.AsyncClient(
            base_url=config.backend_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=config.request_deadline_seconds, trust_env=False)

    async def close(self):
        await self.client.aclose()

    async def models(self):
        response = await self.client.get("/v1/models", timeout=5)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data")
        if payload.get("object") != "list" or not isinstance(records, list):
            raise ValueError("vLLM /v1/models sai schema.")
        by_id = {item.get("id"): item for item in records if isinstance(item, dict)}
        expected = {self.config.base_model, self.config.model_name}
        if set(by_id) != expected:
            raise ValueError(f"Model đang serve không khớp: cần {sorted(expected)}, nhận {sorted(str(k) for k in by_id)}.")
        if not self.config.base_only and by_id[self.config.model_name].get("parent") != self.config.base_model:
            raise ValueError("LoRA model card không trỏ đến đúng backbone; kiểm tra --lora-modules.")
        return payload  # Forward nguyên dữ liệu thật của vLLM, không tự tạo model card.

    async def predict(self, query):
        cfg = self.config
        response = await self.client.post("/v1/chat/completions", json={
            "model": cfg.model_name, "messages": prompt_messages(query.query),
            "max_tokens": cfg.max_new_tokens, "temperature": 0, "top_p": 1,
            "top_k": -1, "repetition_penalty": 1, "seed": cfg.seed,
            "n": 1, "stream": False})
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("vLLM trả content không phải chuỗi.")
        result = competition_result(query.query_id, raw)
        return result, {"raw_output": raw, "finish_reason": choice.get("finish_reason"),
                        "usage": body.get("usage", {})}
