"""Schema BTC; chỉ chuẩn hóa định dạng, không suy luận/sửa đáp án bằng nhãn."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from baseline.metrics import parse_prediction


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=20000)

    @field_validator("query_id", "query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Không được chỉ chứa khoảng trắng.")
        return value


def competition_result(query_id, raw):
    prediction = parse_prediction(raw)
    if prediction is None:
        raise ValueError("Model trả JSON không hợp lệ hoặc thiếu answer/explanation/đơn vị ASCII.")
    answers = [part.strip() for part in prediction["answer"].split(";")]
    units = [part.strip() for part in prediction["unit"].split(";")]
    if len(answers) != len(units) or any(not part for part in answers):
        raise ValueError("Số phần answer và unit không khớp.")
    reasoning = prediction["reasoning"]
    if reasoning is not None:
        reasoning = {"type": reasoning["type"], "steps": reasoning["steps"]}
    return {"query_id": query_id, "answer": "; ".join(answers), "unit": "; ".join(units),
            "explanation": prediction["explanation"].strip(), "reasoning": reasoning}


def validate_response(payload, query_id):
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Response phải là JSON list có đúng một kết quả.")
    result = payload[0]
    required = {"query_id", "answer", "unit", "explanation", "reasoning"}
    if not isinstance(result, dict) or set(result) != required or result["query_id"] != query_id:
        raise ValueError("Response sai fields hoặc query_id.")
    import json
    normalized = competition_result(query_id, json.dumps(result, ensure_ascii=False))
    if normalized != result:
        raise ValueError("Response chưa chuẩn hóa ASCII/'; '/khoảng trắng theo schema.")
    return result
