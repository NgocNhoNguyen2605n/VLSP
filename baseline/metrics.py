"""P1 tạm thời: đáp án và đơn vị đúng ở mọi phần; không giả làm evaluator BTC."""

import ast  # Đọc biểu thức số an toàn, không dùng eval.
import json  # Phân tích JSON do model sinh.
import math  # So sánh số với dung sai công khai.
import operator  # Các phép toán được phép khi đọc đáp án số.
import re  # Nhận dạng số mũ và bỏ code fence ngoài JSON.
import unicodedata  # Chuẩn hóa riêng khi chấm, không sửa dữ liệu train.

UNIT_ALIASES = {"Ω": "ohm", "μ": "u", "µ": "u", "−": "-",
                "³": "^3", "²": "^2", "°C": "degC", "độ": "deg",
                "vòng": "turn"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow}


def unit_ascii(value):
    """Đổi các cách viết đơn vị đã biết sang ASCII; giữ phân biệt m/M, n/N."""
    value = unicodedata.normalize("NFC", value).strip()
    if value in ("-", "—", "–", "lần"):
        return ""
    for old, new in UNIT_ALIASES.items():
        value = value.replace(old, new)
    return re.sub(r"\s+", "", value)


def canonical_text(value):
    """So khớp văn bản sau NFC và gộp khoảng trắng, vẫn giữ chữ hoa/thường."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def numeric_value(value):
    """Chỉ tính biểu thức số nhỏ với pi/sqrt/abs; biểu thức có biến trả None."""
    value = value.strip()
    if len(value) > 256:
        return None
    if "=" in value:
        left, right = value.split("=", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", left.strip()):
            value = right
    value = value.replace("−", "-").replace("×", "*").replace("·", "*")
    value = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda match: "^" + match[0].translate(
        str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")), value)
    value = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", value).replace("^", "**")

    def visit(node):
        """Giới hạn AST để tránh chạy code hoặc biểu thức gây quá tải."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e12):
                raise ValueError("Số mũ quá lớn.")
            return OPERATORS[type(node.op)](left, right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("sqrt", "abs") and len(node.args) == 1 and not node.keywords):
            return {"sqrt": math.sqrt, "abs": abs}[node.func.id](visit(node.args[0]))
        raise ValueError("Không phải biểu thức số được hỗ trợ.")

    try:
        tree = ast.parse(value.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return None
        result = visit(tree.body)
        return result if isinstance(result, (int, float)) and math.isfinite(result) else None
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def parse_prediction(text):
    """Chấp nhận đúng một JSON object; JSON hỏng không được chữa bằng nhãn dev."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if any(not isinstance(result.get(key), str) for key in ("answer", "unit", "explanation")):
        return None
    if not result["answer"].strip() or not result["explanation"].strip():
        return None
    if result.get("reasoning") is not None:
        reasoning = result["reasoning"]
        if (not isinstance(reasoning, dict) or not isinstance(reasoning.get("type"), str)
                or not isinstance(reasoning.get("steps"), list)
                or any(not isinstance(step, str) for step in reasoning["steps"])):
            return None
    # Đây là định dạng output; dữ liệu SFT gốc vẫn được giữ nguyên.
    result["unit"] = "; ".join(unit_ascii(part) for part in result["unit"].split(";"))
    result["reasoning"] = result.get("reasoning")
    return result if result["unit"].isascii() else None


def score_prediction(prediction, gold, rtol=0.05, atol=0.0):
    """All-or-nothing; không cộng điểm từng phần hoặc điểm explanation."""
    if prediction is None:
        return {"exact": False, "tolerant": False, "unsupported_symbolic": False}
    expected = gold["answer"].split(";")
    actual = prediction["answer"].split(";")
    units = prediction["unit"].split(";")
    gold_units = gold["unit"].split(";")
    if (len(actual) != len(expected) or len(units) != len(expected)
            or len(gold_units) != len(expected) or any(not part.strip() for part in actual)):
        return {"exact": False, "tolerant": False, "unsupported_symbolic": False}
    exact, tolerant, unsupported = True, True, False
    for answer, target, unit, target_unit in zip(actual, expected, units, gold_units):
        unit_ok = unit_ascii(unit) == unit_ascii(target_unit)
        text_ok = canonical_text(answer) == canonical_text(target)
        x, y = numeric_value(answer), numeric_value(target)
        number_ok = (x is not None and y is not None
                     and abs(x - y) <= atol + rtol * abs(y))
        exact &= unit_ok and text_ok
        tolerant &= unit_ok and (text_ok or number_ok)
        unsupported |= y is None and bool(re.search(r"[+*/^=]|sqrt\(", target)) and not text_ok
    return {"exact": bool(exact), "tolerant": bool(tolerant),
            "unsupported_symbolic": bool(unsupported)}
