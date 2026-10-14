"""Gọi đúng vLLM self-hosted một lần; không retry, không dùng nhãn test."""

import os

import httpx

from baseline.data import prompt_messages
from baseline.consistency import decoding_settings, select_consistent
from .schema import competition_result


class ModelOutputError(ValueError):
    """Giữ đầu ra model để log khi kiểm tra định dạng thất bại."""

    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


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
        decoding = decoding_settings(cfg)
        payload = {"model": cfg.model_name, "messages": prompt_messages(query.query),
                   "max_tokens": cfg.max_new_tokens, "temperature": decoding["temperature"],
                   "repetition_penalty": 1, "seed": cfg.seed, "stream": False}
        if decoding["beam_search"]:
            # vLLM exposes beam width through n when use_beam_search=true.
            # The returned choices are ranked; the first one is the winner.
            payload.update(use_beam_search=True, n=decoding["beam_width"])
        else:
            payload.update(top_p=decoding["top_p"], top_k=-1, n=decoding["samples"])
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        choices = body["choices"]
        details = {"usage": body.get("usage", {}), "decoding": decoding}
        expected_choices = decoding["beam_width"] if decoding["beam_search"] else decoding["samples"]
        if not isinstance(choices, list) or len(choices) != expected_choices:
            raise ModelOutputError("vLLM không trả đủ số mẫu yêu cầu.", details)
        if any(not isinstance(choice, dict) or not isinstance(choice.get("message"), dict) for choice in choices):
            raise ModelOutputError("vLLM trả choices sai schema.", details)
        if decoding["beam_search"]:
            if len(choices) != decoding["beam_width"]:
                raise ModelOutputError("vLLM không trả đủ beam cho yêu cầu.", details)
            indices = [choice.get("index") for choice in choices]
            if sorted(index for index in indices if type(index) is int) != list(range(len(choices))):
                raise ModelOutputError("vLLM trả index beam trùng hoặc không hợp lệ.", details)
            choices = sorted(choices, key=lambda choice: choice["index"])
            details["beam_search"] = {"beam_width": decoding["beam_width"],
                                       "returned": len(choices), "selected_index": 0}
            choice = choices[0]
        elif decoding["samples"] > 1:
            # vLLM có thể trả choices không theo thứ tự; index quyết định cách xử lý hòa.
            indices = [choice.get("index") for choice in choices]
            if sorted(index for index in indices if type(index) is int) != list(range(len(choices))):
                raise ModelOutputError("vLLM trả index mẫu trùng hoặc không hợp lệ.", details)
            choices = sorted(choices, key=lambda choice: choice["index"])
            candidates = [{"raw_output": choice.get("message", {}).get("content"),
                           "finished": choice.get("finish_reason") == "stop",
                           "finish_reason": choice.get("finish_reason")} for choice in choices]
            selected, consistency = select_consistent(candidates)
            details["self_consistency"] = consistency
            if selected is None:
                raise ModelOutputError("Không có mẫu hoàn tất và hợp lệ trong self-consistency.", details)
            choice = choices[selected]
        else:
            choice = choices[0]
        raw = choice["message"]["content"]
        details.update(raw_output=raw, finish_reason=choice.get("finish_reason"))
        if not isinstance(raw, str):
            raise ModelOutputError("vLLM trả content không phải chuỗi.", details)
        try:
            result = competition_result(query.query_id, raw)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelOutputError(str(error), details) from error
        return result, details
