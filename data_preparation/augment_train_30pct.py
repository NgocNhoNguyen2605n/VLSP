#!/usr/bin/env python3
"""Create a private, checked 30% augmentation of the VLSP physics train CSV.

The generated records combine four controlled operations:
1. parameterized physics templates with recomputed answers;
2. Vietnamese question paraphrasing;
3. alternative, concise COT renderings;
4. equivalent output-unit conventions where safe.

The source CSV is read only. Outputs are written to ./outputs/.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter
from pathlib import Path


SOURCE = Path("/Users/nho/Documents/NCKH /VLSP Dataset/dataset_codex/train_dataset_reviewed.csv")
OUT_DIR = Path("outputs")
SYNTHETIC_OUT = OUT_DIR / "train_dataset_augmented_30pct_synthetic.csv"
COMBINED_OUT = OUT_DIR / "train_dataset_augmented_30pct.csv"
SEED = 20261012
K = 9.0e9


def fmt(x: float, digits: int = 6) -> str:
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{digits}g}"
    return s.replace("e+", "e")


def answer_fmt(x: float) -> str:
    """Format answers in the corpus convention instead of Python e-notation."""
    x = float(x)
    ax = abs(x)
    if ax and (ax >= 1e5 or ax < 1e-3):
        exponent = int(math.floor(math.log10(ax)))
        mantissa = x / (10 ** exponent)
        return f"{mantissa:.2f} * 10^{exponent}"
    return fmt(x)


def fixed(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "cot", "answer", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def paraphrase(rng: random.Random, text: str, kind: str) -> str:
    if kind == "calc":
        options = [
            ("Tính", "Hãy tính"),
            ("Xác định", "Hãy xác định"),
            ("có độ lớn", "có giá trị bằng"),
            ("Hãy tính", "Tính"),
            ("Cho", "Xét"),
        ]
    elif kind == "check":
        options = [
            ("Kiểm tra", "Hãy kiểm tra"),
            ("có phải là", "có đúng là"),
            ("cộng hưởng", "cộng hưởng điện"),
        ]
    else:
        options = [
            ("sẽ thay đổi như thế nào", "thay đổi ra sao"),
            ("Nếu", "Trong trường hợp"),
            ("dòng qua", "cường độ dòng điện qua"),
        ]
    result = text
    # One or two substitutions retain the template semantics.
    rng.shuffle(options)
    for old, new in options[:2]:
        result = result.replace(old, new, 1)
    return result


def cot_style(rng: random.Random, steps: list[str]) -> str:
    styles = [
        lambda xs: "; ".join(xs),
        lambda xs: " → ".join(xs),
        lambda xs: "Ta có " + "; ".join(xs),
        lambda xs: "Theo công thức: " + "; ".join(xs),
    ]
    return rng.choice(styles)(steps)


def answer_with_unit(value: float, base_unit: str, rng: random.Random) -> tuple[str, str, float]:
    """Return answer, unit, and the displayed value; conversions are exact and safe."""
    # Equivalent field units preserve the numerical value.
    if base_unit == "V/m":
        return answer_fmt(value), "V/m", value
    # Energy is converted together with the displayed answer and COT by callers.
    if base_unit == "J" and abs(value) < 2.0 and rng.random() < 0.45:
        return answer_fmt(value * 1000), "mJ", value * 1000
    return answer_fmt(value), base_unit, value


def make_ld(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        if i % 3 == 0:
            q1 = rng.choice([1.2, 1.6, 2.0, 2.4, 3.0, 4.0])
            q2 = rng.choice([1.5, 2.0, 2.5, 3.2, 4.0])
            r_cm = rng.choice([6.0, 8.0, 10.0, 12.0, 15.0])
            r_m = r_cm / 100
            val = K * q1 * 1e-6 * q2 * 1e-6 / r_m**2
            question = f"Hai điện tích q1 = {q1} μC và q2 = {q2} μC đặt cách nhau {r_cm:g} cm trong không khí. {rng.choice(['Tính lực điện giữa chúng.', 'Xác định độ lớn lực tương tác điện.'])}"
            steps = [f"F=k*q1*q2/r^2={fmt(val)} N", "hai điện tích cùng dấu nên lực là lực đẩy"]
            unit = "N"
            ans = answer_fmt(val)
        else:
            q = rng.choice([0.8, 1.2, 1.6, 2.0, 2.6, 3.0])
            a_cm = rng.choice([5.0, 6.0, 8.0, 10.0, 12.0])
            a_m = a_cm / 100
            factor = math.sqrt(2) if i % 3 == 1 else math.sqrt(3)
            angle_text = "tam giác vuông cân" if factor == math.sqrt(2) else "tam giác đều"
            val = factor * K * q * 1e-6 / a_m**2
            question = f"Ba điện tích giống nhau q = {q} μC đặt tại ba đỉnh của {angle_text} cạnh {a_cm:g} cm. Tính cường độ điện trường tổng hợp tại đỉnh góc vuông." if factor == math.sqrt(2) else f"Ba điện tích giống nhau q = {q} μC đặt tại ba đỉnh của {angle_text} cạnh {a_cm:g} cm. Tính cường độ điện trường tổng hợp tại đỉnh cần xét."
            steps = [f"E0=k*q/a^2={fmt(K*q*1e-6/a_m**2)} V/m", f"các vectơ hợp góc {'90' if factor == math.sqrt(2) else '60'} độ", f"E={fmt(val)} V/m"]
            unit = "V/m"
            ans, unit, _ = answer_with_unit(val, unit, rng)
        out.append({"id": f"LD_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": ans, "unit": unit})
    return out


def make_ch(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        if i % 3 == 0:
            R = rng.choice([30, 50, 80, 100, 120])
            f = rng.choice([50, 60, 75, 100])
            L = rng.choice([0.1, 0.2, 0.4, 0.5])
            C = rng.choice([20, 40, 50, 100]) * 1e-6
            omega = 2 * math.pi * f
            xl, xc = omega * L, 1 / (omega * C)
            z = math.sqrt(R * R + (xl - xc) ** 2)
            question = f"Đặt điện áp xoay chiều tần số {f} Hz vào mạch RLC nối tiếp có R = {R} Ω, L = {L} H và C = {C:g} F. Hãy tính tổng trở của mạch."
            steps = [f"X_L=2*pi*f*L={fmt(xl)} ohm", f"X_C=1/(2*pi*f*C)={fmt(xc)} ohm", f"Z=sqrt(R^2+(X_L-X_C)^2)={fmt(z)} ohm"]
            ans, unit, _ = answer_fmt(z), "ohm", z
        elif i % 3 == 1:
            L = rng.choice([0.05, 0.1, 0.2, 0.5])
            C = rng.choice([20, 50, 100, 200]) * 1e-6
            f0 = 1 / (2 * math.pi * math.sqrt(L * C))
            question = f"Mạch RLC có cuộn cảm thuần L = {L} H và tụ điện C = {C:g} F. Xác định tần số cộng hưởng của mạch."
            steps = [f"f0=1/(2*pi*sqrt(L*C))={fmt(f0)} Hz"]
            ans, unit = answer_fmt(f0), "Hz"
        else:
            U = rng.choice([110, 120, 180, 220])
            question = f"Mạch RLC nối tiếp đạt cộng hưởng với điện áp hiệu dụng toàn mạch U = {U} V. Điện áp hiệu dụng trên điện trở bằng bao nhiêu?"
            steps = ["tại cộng hưởng: X_L=X_C nên Z=R", f"U_R=U={U} V"]
            ans, unit = str(U), "V"
        out.append({"id": f"CH_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": ans, "unit": unit})
    return out


def make_nl(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        if i % 3 == 0:
            L = rng.choice([0.05, 0.1, 0.2, 0.5])
            I = rng.choice([1, 2, 3, 4])
            value_j = 0.5 * L * I * I
            ans, unit, displayed = answer_with_unit(value_j, "J", rng)
            question = f"Cuộn cảm có độ tự cảm L = {L} H và dòng điện I = {I} A. Tính năng lượng từ trường tích trữ."
            steps = [f"W_L=L*I^2/2={fmt(value_j)} J", f"đổi sang đơn vị kết quả: {ans} {unit}"]
        elif i % 3 == 1:
            C = rng.choice([10, 20, 40, 50]) * 1e-6
            U = rng.choice([100, 200, 300, 400])
            value_j = 0.5 * C * U * U
            ans, unit, displayed = answer_with_unit(value_j, "J", rng)
            question = f"Tụ điện C = {C:g} F được tích điện đến hiệu điện thế U = {U} V. Tính năng lượng điện trường của tụ."
            steps = [f"W_C=C*U^2/2={fmt(value_j)} J", f"kết quả theo unit yêu cầu: {ans} {unit}"]
        else:
            loss = rng.choice([0.1, 0.2, 0.3, 0.4])
            useful = rng.choice([0.6, 0.8, 0.9, 1.2])
            eta = useful / (useful + loss) * 100
            question = f"Một hệ có năng lượng có ích {useful:g} J và năng lượng hao phí {loss:g} J. Hiệu suất của hệ bằng bao nhiêu?"
            steps = [f"H=W_ich/(W_ich+W_hao)={fmt(eta)}%"]
            ans, unit = answer_fmt(eta), "%"
        out.append({"id": f"NL_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": ans, "unit": unit})
    return out


def make_td(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        C_uF = rng.choice([10, 20, 40, 50, 80])
        U = rng.choice([100, 150, 200, 300, 400])
        C = C_uF * 1e-6
        q = C * U
        if i % 2 == 0:
            question = f"Tụ điện có điện dung {C_uF:g} μF được đặt vào hiệu điện thế {U:g} V. Tính điện tích của tụ."
            steps = [f"Q=C*U={fmt(q)} C", f"Q={answer_fmt(q*1e6)} μC"]
            ans, unit = answer_fmt(q * 1e6), "uC"
        else:
            w = 0.5 * C * U * U
            question = f"Tụ điện C = {C_uF:g} μF tích điện đến hiệu điện thế {U:g} V. Tính năng lượng điện trường."
            ans, unit, _ = answer_with_unit(w, "J", rng)
            steps = [f"W=C*U^2/2={fmt(w)} J", f"kết quả biểu diễn là {ans} {unit}"]
        out.append({"id": f"TD_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": ans, "unit": unit})
    return out


def make_ddt(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        N = rng.choice([100, 200, 400])
        delta_phi = rng.choice([0.002, 0.004, 0.006, 0.01])
        dt = rng.choice([0.02, 0.05, 0.1])
        emf = N * delta_phi / dt
        question = f"Một cuộn dây {N:g} vòng có từ thông qua mỗi vòng biến thiên {delta_phi:g} Wb trong thời gian {dt:g} s. Tính suất điện động cảm ứng trung bình."
        steps = [f"E=N*DeltaPhi/Delta t={fmt(emf)} V", "lấy độ lớn của suất điện động cảm ứng"]
        out.append({"id": f"DDT_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": fmt(emf), "unit": "V"})
    return out


def make_thcb(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    bank = [
        ("Nếu hiệu điện thế hai đầu điện trở không đổi mà điện trở tăng, cường độ dòng điện qua điện trở sẽ thay đổi thế nào?", "Theo định luật Ohm I=U/R; R tăng nên I giảm.", "Điện trở tăng → dòng giảm"),
        ("Khi tăng hiệu điện thế đặt vào một điện trở không đổi, cường độ dòng điện thay đổi ra sao?", "Theo I=U/R; U tăng nên I tăng.", "Dòng điện tăng"),
        ("Trong mạch nối tiếp, nếu một bóng đèn bị đứt dây tóc thì các bóng còn lại hoạt động thế nào?", "Mạch nối tiếp bị hở nên dòng điện trong mạch bằng 0.", "Các bóng còn lại tắt"),
        ("Khi đưa nam châm lại gần cuộn dây kín, trong cuộn dây có xuất hiện dòng điện không?", "Từ thông qua cuộn dây biến thiên nên xuất hiện dòng điện cảm ứng.", "Có"),
        ("Ở vị trí cộng hưởng của mạch RLC nối tiếp, tổng trở của mạch có đặc điểm gì?", "Tại cộng hưởng X_L=X_C nên phần phản kháng triệt tiêu; Z=R.", "Tổng trở nhỏ nhất"),
    ]
    contexts = [
        "trong mạch điện lý tưởng",
        "khi mạch đang ở trạng thái ổn định",
        "bỏ qua ảnh hưởng của dây nối",
        "với các đại lượng khác không đổi",
        "trong điều kiện thí nghiệm thông thường",
        "xét riêng đoạn mạch đang khảo sát",
        "khi nguồn điện được coi là không đổi",
        "trong trường hợp các phần tử làm việc bình thường",
        "ở chế độ xác lập",
        "với giả thiết các thông số không đổi",
        "trong mô hình mạch đơn giản",
        "khi chỉ xét yếu tố được nêu trong đề",
    ]
    out = []
    for i in range(n):
        q, c, a = bank[i % len(bank)]
        q = paraphrase(rng, q, "concept")
        q = q.rstrip("?") + f" {contexts[i % len(contexts)]}?"
        out.append({"id": f"THCB_AUG_{start+i:03d}", "question": q, "cot": c, "answer": a, "unit": "-"})
    return out


def make_dt(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        q_nC = rng.choice([2, 4, 6, 8, 10])
        r_cm = rng.choice([5, 8, 10, 12, 15])
        value = K * q_nC * 1e-9 / (r_cm / 100) ** 2
        base = "V/m" if i % 2 else "N/C"
        alt = "N/C" if base == "V/m" else "V/m"
        question = f"Một điện tích điểm q = {q_nC:g} nC đặt trong không khí. Tính cường độ điện trường tại điểm cách điện tích {r_cm:g} cm."
        steps = [f"E=k*abs(q)/r^2={answer_fmt(value)} V/m", "dùng quy ước unit chuẩn V/m"]
        out.append({"id": f"DT_AUG_{start+i:03d}", "question": paraphrase(rng, question, "calc"), "cot": cot_style(rng, steps), "answer": answer_fmt(value), "unit": "V/m"})
    return out


def make_chlt(rng: random.Random, n: int, start: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        L = rng.choice([0.05, 0.1, 0.2])
        C = rng.choice([20, 50, 100]) * 1e-6
        f0 = 1 / (2 * math.pi * math.sqrt(L * C))
        f = f0 if i % 2 == 0 else f0 * 1.12
        yes = abs(f - f0) < 1e-9
        question = f"Mạch có L = {L:g} H và C = {C:g} F. Kiểm tra tần số f = {f:.2f} Hz có phải là tần số cộng hưởng của mạch không."
        steps = [f"f0=1/(2*pi*sqrt(L*C))={f0:.2f} Hz", f"f {'=' if yes else 'khác'} f0 nên kết luận là {'cộng hưởng' if yes else 'không cộng hưởng'}"]
        out.append({"id": f"CHLT_AUG_{start+i:03d}", "question": paraphrase(rng, question, "check"), "cot": cot_style(rng, steps), "answer": "Có" if yes else "Không", "unit": "-"})
    return out


def quotas(rows: list[dict[str, str]], total: int) -> dict[str, int]:
    prefixes = [re.match(r"^[A-Za-z]+", r["id"]).group() for r in rows]
    counts = Counter(prefixes)
    raw = {p: counts[p] * total / len(rows) for p in counts}
    q = {p: math.floor(v) for p, v in raw.items()}
    remainder = total - sum(q.values())
    for p in sorted(raw, key=lambda x: raw[x] - q[x], reverse=True)[:remainder]:
        q[p] += 1
    return q


def validate(original: list[dict[str, str]], synthetic: list[dict[str, str]], combined: list[dict[str, str]]) -> None:
    assert len(original) == 677, len(original)
    assert len(synthetic) == 203, len(synthetic)
    assert len(combined) == 880, len(combined)
    assert len({r["id"] for r in combined}) == len(combined)
    original_questions = {r["question"].strip() for r in original}
    synthetic_questions = [normalize_question(r["question"]) for r in synthetic]
    assert not original_questions.intersection(synthetic_questions)
    assert len(set(synthetic_questions)) == len(synthetic_questions)
    assert all(r["question"] and r["cot"] and r["answer"] and r["unit"] for r in synthetic)
    assert not any("cấu hình augmentation" in r["question"] for r in synthetic)
    assert not any(r["unit"] == "N/C" for r in synthetic)
    assert not any("e+" in r["answer"].lower() or "e0" in r["answer"].lower() for r in synthetic)
    assert not any("tam giác vuông cân" in r["question"] and "tại đỉnh cần xét" in r["question"] for r in synthetic)


def normalize_question(text: str) -> str:
    """Normalize whitespace and remove legacy auto-generated metadata tags."""
    text = re.sub(r"\s*\(cấu hình augmentation \d+\)", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def generate_unique_block(rng: random.Random, prefix: str, count: int, generator) -> list[dict[str, str]]:
    """Generate a quota of unique questions without synthetic metadata tags."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    attempt = 0
    while len(result) < count and attempt < 30:
        batch_size = max(count * 2, 20)
        candidates = generator(rng, batch_size, attempt * 1000 + 1)
        for row in candidates:
            row["question"] = normalize_question(row["question"])
            if row["question"] in seen:
                continue
            seen.add(row["question"])
            result.append(row)
            if len(result) == count:
                break
        attempt += 1
    if len(result) != count:
        raise RuntimeError(f"Could not generate {count} unique {prefix} questions; got {len(result)}")
    # Re-number IDs after selection so discarded candidates cannot create gaps or collisions.
    for i, row in enumerate(result, 1):
        row["id"] = f"{prefix}_AUG_{i:03d}"
    return result


def main() -> None:
    rng = random.Random(SEED)
    original = read_rows()
    qs = quotas(original, 203)
    generators = {
        "LD": make_ld,
        "CH": make_ch,
        "NL": make_nl,
        "TD": make_td,
        "DDT": make_ddt,
        "THCB": make_thcb,
        "DT": make_dt,
        "CHLT": make_chlt,
    }
    synthetic = []
    for prefix in ["LD", "CH", "NL", "TD", "DDT", "THCB", "DT", "CHLT"]:
        synthetic.extend(generate_unique_block(rng, prefix, qs[prefix], generators[prefix]))
    # Shuffle only the synthetic block; original row order remains untouched.
    rng.shuffle(synthetic)
    combined = original + synthetic
    validate(original, synthetic, combined)
    write_rows(SYNTHETIC_OUT, synthetic)
    write_rows(COMBINED_OUT, combined)
    print(f"original={len(original)} synthetic={len(synthetic)} combined={len(combined)}")
    print("quotas=" + repr(qs))
    print("synthetic_units=" + repr(Counter(r["unit"] for r in synthetic)))
    print("synthetic_prefixes=" + repr(Counter(re.match(r"^[A-Za-z]+", r["id"]).group() for r in synthetic)))
    print(f"wrote={SYNTHETIC_OUT}")
    print(f"wrote={COMBINED_OUT}")


if __name__ == "__main__":
    main()
