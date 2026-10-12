"""API BTC: POST /predict, GET /v1/models (proxy thật), GET /healthz."""

import argparse
import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from .backend import VLLMBackend
from .config import ServingConfig
from .schema import Query


def create_app(config=None, backend=None):
    cfg = config or ServingConfig.load()
    engine = backend or VLLMBackend(cfg)
    lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if backend is None:
            cfg.require_prepared()
        try:
            await engine.models()  # Dừng khởi động nếu đang serve nhầm base/adapter.
            yield
        finally:
            await engine.close()

    app = FastAPI(title="Viper ViPRA 2026", lifespan=lifespan, docs_url=None, redoc_url=None)

    def record(value, filename="requests.jsonl"):
        folder = Path(cfg.log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    @app.get("/v1/models")
    async def models():
        try:
            return await engine.models()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, "vLLM chưa sẵn sàng hoặc model không khớp.") from error

    @app.get("/healthz")
    async def health():
        await models()
        return {"status": "ready", "model": cfg.model_name, "team": cfg.team_name,
                "backbone_precision": cfg.backbone_precision}

    @app.post("/predict")
    async def predict(query: Query):
        start = time.perf_counter()
        event = {"query_id": query.query_id, "query": query.query, "time_unix": time.time()}
        record(dict(event, status="processing"), "activity.jsonl")
        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                async with lock:
                    result, details = await engine.predict(query)
            event.update(status="ok", prediction=result, **details)
            return [result]
        except (TimeoutError, httpx.TimeoutException) as error:
            event.update(status="timeout", error=type(error).__name__)
            raise HTTPException(504, "Quá deadline nội bộ; không retry model.") from error
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            event.update(status="error", error=str(error))
            raise HTTPException(502, "Model không trả kết quả hợp lệ; xem requests.jsonl.") from error
        finally:
            event["seconds"] = time.perf_counter() - start
            record(event)
            record(event, "activity.jsonl")

    return app


def main():
    parser = argparse.ArgumentParser(description="API ViPRA, cần vLLM đang chạy")
    parser.add_argument("--config", default="deployment/serving.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    os.environ["VIPRA_SERVING_CONFIG"] = str(Path(args.config).resolve())
    import uvicorn
    uvicorn.run("submission.app:create_app", factory=True, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
