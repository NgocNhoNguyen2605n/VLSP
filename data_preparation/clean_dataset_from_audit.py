#!/usr/bin/env python3
"""Clean train + private augmentation using the supplied audit report.

The original train, synthetic augmentation and dev files are read-only inputs.
This script writes new cleaned copies under outputs/cleaned_dataset_20261012/.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path


TRAIN = Path("/Users/nho/Documents/NCKH /VLSP Dataset/dataset_codex/train_dataset_reviewed.csv")
SYNTHETIC = Path("/Users/nho/Documents/ChatGPT/VLSP/outputs/train_dataset_augmented_30pct_synthetic.csv")
DEV = Path("/Users/nho/Documents/ChatGPT/VLSP/source-main/dev_dataset.csv")
OUT = Path("/Users/nho/Documents/ChatGPT/VLSP/outputs/cleaned_dataset_20261012_v2")
FIELDS = ["id", "question", "cot", "answer", "unit"]
K = 9e9


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and list(rows[0]) == FIELDS, (path, list(rows[0]))
    assert all(all(v is not None for v in r.values()) for r in rows)
    return rows


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def answer_scientific(text: str) -> str:
    """Convert Python-style e notation to the corpus notation."""
    m = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[eE]([+-]?\d+)\s*", text)
    if not m:
        return text.strip()
    mantissa = float(m.group(1))
    exponent = int(m.group(2))
    return f"{mantissa:g} * 10^{exponent}"


def clean_common(row: dict[str, str]) -> None:
    row["question"] = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", row["question"])
    row["question"] = re.sub(r"\s+", " ", row["question"]).strip()
    row["cot"] = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", row["cot"])
    row["cot"] = re.sub(r"\s+", " ", row["cot"]).strip()
    row["answer"] = answer_scientific(row["answer"])
    if row["unit"] == "N/C":
        row["unit"] = "V/m"
        row["cot"] = row["cot"].replace("N/C", "V/m")


def triangle_question(q: str) -> str:
    q = re.sub(
        r"tam giác vuông cân cạnh (?:a\s*=\s*)?([0-9]+(?:\.[0-9]+)?) (cm|m)",
        r"tam giác vuông cân có hai cạnh góc vuông đều dài \1 \2",
        q,
        flags=re.I,
    )
    if "Ba điện tích" in q and "cường độ điện trường" in q:
        if "tam giác vuông cân" in q and "tại đỉnh cần xét" in q:
            q = q.replace("tại đỉnh cần xét", "tại đỉnh góc vuông")
        if "tại vị trí q3" in q and "do q1 và q2" not in q:
            q = q.replace("cường độ điện trường tổng hợp tại vị trí q3", "cường độ điện trường do q1 và q2 gây ra tại vị trí q3")
        elif "tại đỉnh góc vuông" in q and "do hai điện tích còn lại" not in q:
            q = q.replace("cường độ điện trường tổng hợp tại đỉnh góc vuông", "cường độ điện trường do hai điện tích còn lại gây ra tại đỉnh góc vuông")
        elif "tại đỉnh cần xét" in q:
            q = q.replace("cường độ điện trường tổng hợp tại đỉnh cần xét", "cường độ điện trường do hai điện tích còn lại gây ra tại đỉnh đang xét")
    return q


def update(row: dict[str, str], question: str, cot: str, answer: str | None = None, unit: str | None = None) -> None:
    row["question"] = question
    row["cot"] = cot
    if answer is not None:
        row["answer"] = answer
    if unit is not None:
        row["unit"] = unit


def repair_high(row: dict[str, str]) -> None:
    rid = row["id"]
    if rid == "CH142":
        update(row,
            "Một đoạn mạch RLC nối tiếp gồm điện trở R, tụ điện C và cuộn cảm L. Đặt điện áp xoay chiều hiệu dụng 100 V vào hai đầu mạch. Biết điện áp hiệu dụng giữa hai đầu đoạn R–C và đoạn R–L đều bằng 120 V, đồng thời mạch đang ở trạng thái cộng hưởng. Tính điện áp hiệu dụng giữa hai đầu tụ điện C.",
            "Tại cộng hưởng U_R=U=100 V. U_C=sqrt(U_RC^2-U_R^2)=sqrt(120^2-100^2)=66.30 V.", "66.30", "V")
    elif rid == "CH145":
        update(row,
            "Một đoạn mạch RLC nối tiếp được mắc vào nguồn xoay chiều hiệu dụng 80 V. Trong điều kiện cộng hưởng, điện áp hiệu dụng giữa hai đầu đoạn R–C và đoạn R–L đều bằng 100 V. Tính điện áp hiệu dụng giữa hai đầu tụ điện C.",
            "Tại cộng hưởng U_R=U=80 V. U_C=sqrt(U_RC^2-U_R^2)=sqrt(100^2-80^2)=60 V.", "60", "V")
    elif rid == "LD111":
        update(row,
            "Hai điện tích q1 = -1 * 10^-8 C, q2 = -6 * 10^-8 C được đặt tại A và B cách nhau 10 cm trong không khí. Điện tích q3 = -7 * 10^-8 C đặt tại C, biết CA = 6 cm và CB = 8 cm. Tính lực điện tổng hợp tác dụng lên q3.",
            "AB=10 cm, CA=6 cm, CB=8 cm nên tam giác tại C vuông. F1=9e9*7e-16/0.06^2=1.75e-3 N; F2=5.906e-3 N; F=sqrt(F1^2+F2^2)=6.16e-3 N.", "6.16 * 10^-3", "N")
    elif rid == "DDT340":
        update(row,
            "Mạch gồm R = 12 Ω mắc nối tiếp với tụ điện C = 80 μF, hoạt động ở tần số f = 60 Hz. Tính dung kháng của tụ điện.",
            "X_C=1/(2*pi*f*C)=1/(2*pi*60*80e-6)=33.16 ohm.", "33.16", "ohm")
    elif rid == "CH244":
        update(row,
            "Cho mạch A_B với R1 = 40 Ω, R2 = 25 Ω thỏa mãn LCω^2 = 1 và uAM vuông pha uMB. Nếu đặt điện áp hiệu dụng U = 90 V vào đoạn M_B gồm R2 và L, tính công suất tiêu thụ trên đoạn M_B.",
            "Từ điều kiện cộng hưởng và vuông pha X_L^2=R1*R2=1000. P_MB=U^2*R2/(R2^2+X_L^2)=90^2*25/(25^2+1000)=124.62 W.", "124.62", "W")
    elif rid in {"CH085", "CH098"}:
        f = "1000" if rid == "CH085" else "2000"
        ans = "5.07" if rid == "CH085" else "1.27"
        update(row, f"Một mạch cần cộng hưởng tại {f} Hz, cuộn dây có L = 0.005 H. Tính điện dung C của tụ điện.",
               f"C=1/(4*pi^2*f^2*L)={ans} uF.", ans, "uF")
    elif rid == "LD045":
        update(row, "Hai điện tích q1 = +1 μC, q2 = -4 μC đặt cách nhau 6 cm. Điểm M nằm trên đoạn AB cách A 2 cm và đặt điện tích thử q0 = +1 μC tại M. Tính độ lớn lực tổng hợp tác dụng lên q0.",
               "F=9e9*(1e-6/0.02^2+4e-6/0.04^2)*1e-6=45 N.")
    elif rid == "LD048":
        update(row, "Một điện tích q1 = +2 μC tại A và q2 = -2 μC tại B cách nhau 10 cm. Đặt điện tích thử q0 = +1 μC tại trung điểm AB. Tính lực tổng hợp tác dụng lên q0.",
               "Tại trung điểm, hai lực cùng chiều: F=2*k*abs(2e-6*1e-6)/0.05^2=14.4 N.")
    elif rid == "LD049":
        update(row, "Hai điện tích q1 = +5 μC, q2 = +5 μC đặt tại A và B cách nhau 10 cm. Đặt điện tích thử q0 = +1 μC tại điểm cách A 6 cm và B 8 cm. Tính độ lớn lực tổng hợp tác dụng lên q0.",
               "E1=1.25e7 V/m, E2=7.03125e6 V/m và hai điện trường vuông góc; F=sqrt(E1^2+E2^2)*1e-6=14.34 N.")
    elif rid == "LD002":
        update(row, "Ba điện tích được đặt tại ba đỉnh của tam giác vuông ABC, góc vuông tại B, với AB = 4 m và BC = 5 m. Các điện tích lần lượt là qA = 5.0 μC, qB = -5.0 μC, qC = 4.0 μC. Tìm lực điện tổng hợp tác dụng lên điện tích tại A.",
               "Góc vuông tại B nên AC=sqrt(4^2+5^2). F_BA=1.406e-2 N, F_CA=4.39e-3 N; cộng thành phần cho F_A=1.18e-2 N.")
    elif rid == "LD021":
        update(row, "Một điện tích q = -1 μC bị hút bởi hai điện tích +1 μC nằm trên cùng đường thẳng, ở hai phía của q, cách q lần lượt 5 cm và 12 cm. Tính độ lớn lực tổng hợp tác dụng lên q.",
               "Hai lực hút ngược chiều. F=abs(3.6-0.625)=2.98 N.")
    elif rid == "LD024":
        update(row, "Ba điện tích q1 = +1 μC, q2 = -2 μC và q3 = +3 μC đặt tại ba đỉnh của tam giác vuông cân; q3 ở đỉnh góc vuông và hai cạnh góc vuông đều dài 0.1 m. Tính lực tổng hợp tác dụng lên q3.",
               "F13=2.7 N, F23=5.4 N; hai lực vuông góc nên F=sqrt(2.7^2+5.4^2)=6.04 N.")
    elif rid == "LD050":
        update(row, "Hai điện tích q1 = +1 μC, q2 = -1 μC đặt cách nhau 10 cm. Điện tích thử q0 = +2 μC đặt tại trung điểm AB. Tính độ lớn lực điện tổng hợp tác dụng lên q0.",
               "Hai lực cùng chiều: F=2*9e9*1e-6*2e-6/0.05^2=14.4 N.")
    elif rid == "LD342":
        update(row, "Hai điện tích q1 = 3.6 * 10^-6 C, q2 = 3.0 * 10^-6 C đặt tại A và B cách nhau 9.2 cm. Điểm M nằm trên phần kéo dài AB về phía B, cách A 14.2 cm. Tính cường độ điện trường tổng hợp tại M.",
               "MB=MA-AB=0.050 m. E=9e9*(3.6e-6/0.142^2+3.0e-6/0.05^2)=1.24e7 V/m.")
    elif rid == "LD355":
        update(row, "Hai điện tích q1 = 4.2 * 10^-6 C, q2 = -1.2 * 10^-6 C đặt tại A và B cách nhau 7.5 cm. Điểm M nằm trên phần kéo dài AB về phía A, cách A 10.8 cm. Tính cường độ điện trường tổng hợp tại M.",
               "MB=MA+AB=0.183 m. E=abs(9e9*4.2e-6/0.108^2-9e9*1.2e-6/0.183^2)=2.92e6 V/m.")
    elif rid == "LD357":
        update(row, "Hai điện tích q1 = 3.6 * 10^-6 C, q2 = -3.0 * 10^-6 C đặt tại A và B cách nhau 9.2 cm. Điểm M nằm giữa A và B, cách A 4.4 cm. Tính cường độ điện trường tổng hợp tại M.",
               "MB=0.048 m. Hai điện trường cùng chiều nên E=9e9*(3.6e-6/0.044^2+3.0e-6/0.048^2)=2.85e7 V/m.")
    elif rid == "DT025":
        update(row, "Cho q1 = -9 * 10^-6 C tại O và q2 = 4 * 10^-6 C tại tọa độ +20 cm trên trục Ox. Tìm tọa độ điểm trên Ox tại đó cường độ điện trường bằng không.",
               "3/x=2/(x-20), suy ra x=60 cm.")
    elif rid in {"CH237", "CH240"}:
        if rid == "CH237":
            r1, r2, u, xl2, ans = "40", "20", "90", "800", "51.96"
        else:
            r1, r2, u, xl2, ans = "50", "10", "75", "500", "30.62"
        update(row, f"Mạch A_B có R1 = {r1} Ω nối tiếp C và R2 = {r2} Ω nối tiếp L, thỏa mãn LCω^2 = 1. Biết uAM vuông pha uMB. Đặt điện áp hiệu dụng U = {u} V lên A_B. Tính điện áp hiệu dụng trên đoạn M_B.",
               f"X_L^2=R1*R2={xl2}. I={u}/({r1}+{r2})={float(u)/(float(r1)+float(r2)):.2f} A; U_MB={ans} V.", ans, "V")
    elif rid == "TD381":
        update(row, "Tụ C1 = 5 μF được nạp đến 10 V. Cắt tụ khỏi nguồn rồi mắc song song với tụ C2 = 5 μF chưa tích điện. Tính năng lượng của hệ sau khi ghép.",
               "Q=C1*U=5e-5 C; C_total=C1+C2=10 uF; U=Q/C_total=5 V; W=Q^2/(2*C_total)=125 uJ.")
    elif rid == "TD391":
        update(row, "Một tụ điện phẳng có điện dung C1 = 88.5 pF, được tích điện ở U = 200 V. Khi vẫn nối với nguồn, khoảng cách giữa hai bản tăng gấp đôi. Tính công của nguồn trong quá trình này.",
               "C2=C1/2. Công của nguồn A=U^2*(C2-C1)=-C1*U^2/2=-1.77 uJ.")


def repair_medium(row: dict[str, str]) -> None:
    rid = row["id"]
    if rid in {"THCB002", "THCB102"}:
        row["question"] += " Giả sử sai số dụng cụ bằng đúng độ chia nhỏ nhất."
        row["cot"] = row["cot"].replace("Theo quy ước Delta=ĐCNN đang dùng trong nhãn: ", "Sai số dụng cụ bằng ĐCNN: ")
    elif rid == "CHLT006":
        row["question"] = row["question"].replace("tại tần số 80 Hz", "tại tần số 79.58 Hz")
        row["cot"] = row["cot"].replace("≈ 79.58 Hz", "=79.58 Hz")
    elif rid in {"NL365", "NL376"}:
        row["question"] = row["question"].replace("hiệu điện thế tức thời", "độ lớn hiệu điện thế tức thời")
    elif rid == "LD346":
        row["question"] = "Hai điện tích q1=4.2*10^-6 C và q2=4.8*10^-6 C đặt tại A và B, cách nhau 6.0 cm. Điểm M nằm trên đường thẳng AB, ngoài đoạn AB về phía A, cách A 8.0 cm. Tính cường độ điện trường tổng hợp tại M."


def repair_dev_overlap_train(row: dict[str, str]) -> None:
    """Replace train-side overlaps with new valid instances; dev is never written."""
    rid = row["id"]
    if rid == "LD066":
        update(row, "Hai điện tích điểm q1=+2*10^-2 μC và q2=-2*10^-2 μC đặt tại A,B cách nhau a=24 cm. Điểm M nằm trên đường trung trực của AB và MA=MB=a. Tính cường độ điện trường tại M.", "r=sqrt(a^2-(a/2)^2)=0.2078 m. E=k*q*a/r^3=3125 V/m.", "3125", "V/m")
    elif rid == "LD211":
        ans = math.sqrt(2) * K * (4e-6)**2 / 0.12**2
        update(row, "Ba điện tích giống nhau q=+4*10^-6 C đặt tại ba đỉnh của tam giác vuông cân có hai cạnh góc vuông đều dài 12 cm. Tính lực tổng hợp tác dụng lên điện tích tại góc vuông.", "F0=k*q^2/a^2=10 N. Hai lực vuông góc nên F=sqrt(2)*F0=14.14 N.", "14.14", "N")
    elif rid == "LD227":
        ans = math.sqrt(2) * K * (4e-8)**2 / 0.18**2
        update(row, "Ba điện tích giống nhau q=-4*10^-8 C đặt tại ba đỉnh của tam giác vuông cân có hai cạnh góc vuông đều dài 18 cm. Tính lực tổng hợp tác dụng lên điện tích tại góc vuông.", "F0=k*q^2/a^2=4.444e-4 N. Hai lực vuông góc nên F=sqrt(2)*F0=6.29e-4 N.", "6.29 * 10^-4", "N")
    elif rid == "LD329":
        val = math.sqrt(3) * K * 1.4e-6 / 0.11**2
        update(row, "Ba điện tích q1=q2=q3=1.4*10^-6 C đặt tại 3 đỉnh của tam giác đều cạnh 11 cm. Tính cường độ điện trường do q1 và q2 gây ra tại vị trí q3.", "E0=k*q/a^2=1.04132e6 V/m. Hai vectơ hợp góc 60 độ nên E=sqrt(3)*E0=%.3g V/m." % val, "%.3g * 10^0" % val if val < 1e5 else f"{val/1e6:.3g} * 10^6", "V/m")
    elif rid == "LD335":
        val = math.sqrt(2) * K * 1.5e-6 / 0.064**2
        update(row, "Ba điện tích giống nhau q=1.5*10^-6 C đặt tại 3 đỉnh của tam giác vuông cân có hai cạnh góc vuông đều dài 6.4 cm. Tính cường độ điện trường do hai điện tích còn lại gây ra tại đỉnh góc vuông.", "E0=k*q/a^2=3.2959e6 V/m. Hai vectơ vuông góc nên E=sqrt(2)*E0=%.3g V/m." % val, f"{val/1e6:.3g} * 10^6", "V/m")


def repair_field_wording(row: dict[str, str]) -> None:
    q = row["question"]
    if row["id"] == "LD084":
        return
    if "Ba điện tích" not in q or "cường độ điện trường" not in q:
        return
    q = triangle_question(q)
    row["question"] = q


def triangle_field(q: float, a_cm: float, factor: float) -> tuple[str, str]:
    value = factor * K * q * 1e-6 / (a_cm / 100) ** 2
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} * 10^{exponent}", f"E0=k*q/a^2={K*q*1e-6/(a_cm/100)**2:.6g} V/m; E={value:.6g} V/m."


def replace_original_duplicates(rows: list[dict[str, str]]) -> None:
    by_id = {r["id"]: r for r in rows}
    ans, cot = triangle_field(2.6, 11.1, math.sqrt(2))
    update(by_id["LD309"], "Ba điện tích giống nhau q = 2.6 * 10^-6 C đặt tại 3 đỉnh của tam giác vuông cân có hai cạnh góc vuông đều dài 11.1 cm. Tính cường độ điện trường do hai điện tích còn lại gây ra tại đỉnh góc vuông.", cot, ans, "V/m")
    ans, cot = triangle_field(2.6, 11.2, math.sqrt(3))
    update(by_id["LD305"], "Ba điện tích q1=q2=q3=2.6 * 10^-6 C đặt tại 3 đỉnh của tam giác đều cạnh 11.2 cm. Tính cường độ điện trường do q1 và q2 gây ra tại vị trí q3.", cot, ans, "V/m")
    ans, cot = triangle_field(1.2, 5.8, math.sqrt(2))
    update(by_id["LD299"], "Ba điện tích giống nhau q = 1.2 * 10^-6 C đặt tại 3 đỉnh của tam giác vuông cân có hai cạnh góc vuông đều dài 5.8 cm. Tính cường độ điện trường do hai điện tích còn lại gây ra tại đỉnh góc vuông.", cot, ans, "V/m")
    update(by_id["LD059"], "Đặt 4 điện tích có cùng độ lớn q tại 4 đỉnh của hình vuông ABCD cạnh 12 cm; điện tích dương ở A,C và điện tích âm ở B,D. Xác định cường độ điện trường tại giao điểm hai đường chéo.", "Tại tâm, các cặp điện tích đối diện tạo điện trường bằng nhau và ngược chiều; tổng E=0.", "0", "V/m")
    update(by_id["CH151"], "Đặt điện áp u=200*sqrt(2)*cos(100*pi*t) V vào mạch RLC nối tiếp có R=120 Ω, L=1/pi H và C=10^-4/(2*pi) F. Tính tổng trở Z.", "X_L=100 ohm; X_C=200 ohm; Z=sqrt(120^2+(100-200)^2)=156.20 ohm.", "156.20", "ohm")
    value = math.sqrt(3) * K * 5e-16 / 0.09**2
    update(by_id["DT051"], "Hai điện tích q1=q2=5 * 10^-16 C đặt tại B,C của tam giác đều ABC cạnh 9 cm. Tính cường độ điện trường do q1 và q2 gây ra tại A.", "Hai điện trường hợp góc 60 độ: E=sqrt(3)*k*q/a^2=%.6g V/m." % value, f"{value:.3g}", "V/m")


def new_synthetic_variant(row_id: str, serial: int, prefix: str) -> dict[str, str]:
    """Replace reported repeated synthetic physics instances with new parameters."""
    if prefix == "THCB":
        bank = [
            ("Trong mạch song song lý tưởng, hiệu điện thế hai nhánh có quan hệ thế nào?", "Hai nhánh mắc song song có cùng hiệu điện thế.", "Hiệu điện thế hai nhánh bằng nhau"),
            ("Trong mạch nối tiếp lý tưởng, cường độ dòng điện qua các phần tử có quan hệ thế nào?", "Mạch nối tiếp chỉ có một dòng điện nên cường độ qua các phần tử bằng nhau.", "Cường độ dòng điện bằng nhau"),
            ("Khi chiều dài dây dẫn tăng còn vật liệu và tiết diện không đổi, điện trở thay đổi ra sao?", "R=rho*l/S; chiều dài tăng nên điện trở tăng.", "Điện trở tăng"),
            ("Khi tiết diện dây dẫn tăng còn vật liệu và chiều dài không đổi, điện trở thay đổi ra sao?", "R=rho*l/S; tiết diện tăng nên điện trở giảm.", "Điện trở giảm"),
            ("Khi vật chuyển động thẳng đều, hợp lực tác dụng lên vật bằng bao nhiêu?", "Vận tốc không đổi nên gia tốc bằng 0; theo định luật II Newton hợp lực bằng 0.", "Bằng 0"),
            ("Khi tăng khối lượng của vật nhưng giữ nguyên lực tác dụng, gia tốc của vật thay đổi thế nào?", "Theo a=F/m, khối lượng tăng thì gia tốc giảm.", "Gia tốc giảm"),
            ("Một vật rơi tự do gần mặt đất, nếu bỏ qua lực cản không khí thì gia tốc có đặc điểm gì?", "Vật rơi tự do có gia tốc gần bằng gia tốc trọng trường và hướng xuống dưới.", "Gia tốc gần bằng g"),
            ("Khi tăng biên độ dao động của con lắc trong giới hạn góc nhỏ, chu kỳ thay đổi thế nào?", "Trong gần đúng góc nhỏ, chu kỳ con lắc không phụ thuộc biên độ.", "Chu kỳ không đổi"),
            ("Trong phản xạ ánh sáng, góc phản xạ có quan hệ thế nào với góc tới?", "Theo định luật phản xạ ánh sáng, góc phản xạ bằng góc tới.", "Bằng góc tới"),
            ("Khi tần số của sóng tăng còn vận tốc truyền sóng không đổi, bước sóng thay đổi ra sao?", "Theo v=lambda*f, bước sóng giảm khi tần số tăng.", "Bước sóng giảm"),
            ("Nếu lực tác dụng vuông góc với độ dời của vật thì công của lực bằng bao nhiêu?", "A=F*s*cos(90 độ)=0.", "Bằng 0"),
            ("Khi nhiệt độ của một lượng khí tăng trong bình kín có thể tích không đổi, áp suất thay đổi ra sao?", "Với thể tích không đổi, áp suất tỉ lệ thuận với nhiệt độ tuyệt đối.", "Áp suất tăng"),
        ]
        q, c, a = bank[serial % len(bank)]
        return {"id": row_id, "question": q, "cot": c, "answer": a, "unit": "-"}
    if prefix == "DT":
        qn, rc = [(3, 7), (5, 9), (7, 11), (9, 13), (4, 14)][serial % 5]
        val = K * qn * 1e-9 / (rc / 100) ** 2
        return {"id": row_id, "question": f"Một điện tích điểm q={qn} nC đặt trong không khí. Tính cường độ điện trường tại điểm cách điện tích {rc} cm.", "cot": f"E=k*q/r^2={val:.6g} V/m.", "answer": f"{val:.6g}", "unit": "V/m"}
    if prefix == "DDT":
        n, phi, dt = [(150,.003,.04),(250,.005,.08),(350,.007,.06),(450,.009,.12),(550,.011,.1)][serial % 5]
        val=n*phi/dt
        return {"id":row_id,"question":f"Cuộn dây {n} vòng có từ thông qua mỗi vòng biến thiên {phi:g} Wb trong {dt:g} s. Tính suất điện động cảm ứng trung bình.","cot":f"E=N*DeltaPhi/Delta t={val:g} V.","answer":f"{val:g}","unit":"V"}
    if prefix == "NL":
        typ=serial%3
        if typ==0:
            L,I=[(.08,2.5),(.12,3.5),(.3,1.5),(.45,2.5)][serial%4]; val=.5*L*I*I
            return {"id":row_id,"question":f"Cuộn cảm có L={L:g} H và dòng điện I={I:g} A. Tính năng lượng từ trường.","cot":f"W=L*I^2/2={val:g} J.","answer":f"{val:g}","unit":"J"}
        if typ==1:
            C,U=[(15,250),(25,350),(35,450),(60,150)][serial%4]; val=.5*C*1e-6*U*U
            return {"id":row_id,"question":f"Tụ điện C={C:g} uF được tích điện đến U={U:g} V. Tính năng lượng điện trường.","cot":f"W=C*U^2/2={val:g} J.","answer":f"{val:g}","unit":"J"}
        loss,useful=[(.15,.75),(.25,1.05),(.35,1.4),(.45,1.8)][serial%4]; val=100*useful/(loss+useful)
        return {"id":row_id,"question":f"Một hệ có năng lượng có ích {useful:g} J và hao phí {loss:g} J. Tính hiệu suất.","cot":f"H=W_ich/(W_ich+W_hao)*100%={val:g}%.","answer":f"{val:g}","unit":"%"}
    if prefix == "TD":
        C,U=[(12,180),(18,260),(28,320),(45,240),(70,180)][serial%5]
        val=C*1e-6*U
        return {"id":row_id,"question":f"Tụ điện có điện dung {C:g} uF đặt vào hiệu điện thế {U:g} V. Tính điện tích của tụ.","cot":f"Q=C*U={val:g} C={val*1e6:g} uC.","answer":f"{val*1e6:g}","unit":"uC"}
    if prefix == "CH":
        L,C=[(.07,30e-6),(.15,70e-6),(.25,120e-6),(.35,180e-6),(.6,40e-6)][serial%5]
        val=1/(2*math.pi*math.sqrt(L*C))
        return {"id":row_id,"question":f"Mạch RLC có cuộn cảm L={L:g} H và tụ điện C={C:g} F. Tính tần số cộng hưởng.","cot":f"f0=1/(2*pi*sqrt(L*C))={val:g} Hz.","answer":f"{val:g}","unit":"Hz"}
    # LD replacement: always a fresh triangle-field parameter set.
    q,a=[(.9,5.5),(1.1,7.5),(1.3,9.5),(1.7,11.5),(2.1,13.5),(2.9,15.5)][serial%6]
    factor=math.sqrt(2) if serial%2==0 else math.sqrt(3)
    shape="tam giác vuông cân có hai cạnh góc vuông" if factor==math.sqrt(2) else "tam giác đều cạnh"
    target="tại đỉnh góc vuông" if factor==math.sqrt(2) else "tại vị trí q3"
    val=factor*K*q*1e-6/(a/100)**2
    return {"id":row_id,"question":f"Ba điện tích giống nhau q={q:g} uC đặt tại ba đỉnh của {shape} đều dài {a:g} cm. Tính cường độ điện trường do hai điện tích còn lại gây ra {target}.","cot":f"E0=k*q/a^2={K*q*1e-6/(a/100)**2:.6g} V/m; E={val:.6g} V/m.","answer":f"{val:.6g}","unit":"V/m"}


REPEATED_SYNTHETIC = {
    "THCB_AUG_003","DT_AUG_011","DDT_AUG_020","DDT_AUG_005","NL_AUG_023","THCB_AUG_011","THCB_AUG_006","TD_AUG_009","DDT_AUG_015","LD_AUG_028","CH_AUG_013","TD_AUG_022","THCB_AUG_010","THCB_AUG_002","THCB_AUG_007","NL_AUG_017","TD_AUG_005","LD_AUG_008","DDT_AUG_006","TD_AUG_026","LD_AUG_034","LD_AUG_055","LD_AUG_047","CH_AUG_034","THCB_AUG_009","CH_AUG_005","NL_AUG_007"
}


def main() -> None:
    train = read(TRAIN)
    synthetic = read(SYNTHETIC)
    dev_hash_before = sha256(DEV)
    assert len(train) == 677 and len(synthetic) == 203

    for r in train + synthetic:
        clean_common(r)
        r["question"] = triangle_question(r["question"])
        repair_field_wording(r)
        repair_high(r)
        repair_medium(r)
        repair_dev_overlap_train(r)

    # Remove implementation/history references from all augmented CoTs.
    for r in synthetic:
        r["cot"] = re.sub(r"(?:Theo )?(?:nhãn gốc|cấu hình nhãn gốc|COT gốc|cấu hình chuẩn của nhãn|quy ước unit chuẩn)", "", r["cot"], flags=re.I)
        r["cot"] = r["cot"].replace("kết quả theo unit yêu cầu", "đổi đơn vị").replace("kết quả biểu diễn là", "đổi đơn vị")
        r["cot"] = re.sub(r"\s+", " ", r["cot"]).strip(" ;")

    # Remaining wording repairs identified by the second clean audit.
    for r in synthetic:
        if r["id"] == "THCB_AUG_004":
            r["question"] = "Khi đưa nam châm lại gần cuộn dây kín, từ thông qua cuộn dây biến thiên. Trong cuộn dây có xuất hiện dòng điện không?"
        elif r["id"] in {"LD_AUG_034", "LD_AUG_047"}:
            r["question"] = r["question"].replace("tam giác đều cạnh đều dài", "tam giác đều có cạnh dài")

    # Replace semantic duplicates from the audit with new valid instances in the same ID group.
    counters = Counter()
    for r in synthetic:
        if r["id"] in REPEATED_SYNTHETIC:
            prefix = re.match(r"^[A-Za-z]+", r["id"]).group()
            replacement = new_synthetic_variant(r["id"], counters[prefix], prefix)
            counters[prefix] += 1
            r.update(replacement)

    replace_original_duplicates(train)

    # Apply final normalization once more after all targeted edits.
    for r in train + synthetic:
        clean_common(r)

    combined = train + synthetic
    assert sha256(DEV) == dev_hash_before, "Dev file changed unexpectedly"
    assert len(train) == 677 and len(synthetic) == 203 and len(combined) == 880
    assert len({r["id"] for r in combined}) == 880
    assert all(list(r) == FIELDS and all(v != "" for v in r.values()) for r in combined)
    assert not any(r["unit"] == "N/C" for r in combined)
    assert not any("cấu hình augmentation" in r["question"] for r in combined)
    assert not any("theo nhãn gốc" in r["cot"].lower() or "cot gốc" in r["cot"].lower() for r in combined)
    assert not any(re.search(r"tam giác vuông cân cạnh\s+[0-9]", r["question"], re.I) for r in combined)
    assert not any("tại đỉnh cần xét" in r["question"] for r in combined)

    # Exact question duplicates are removed while preserving row counts; repaired replacements remain in-group.
    assert len({r["question"] for r in combined}) == len(combined)
    write(OUT / "train_dataset_reviewed_clean.csv", train)
    write(OUT / "train_dataset_augmented_30pct_synthetic_clean.csv", synthetic)
    write(OUT / "train_dataset_augmented_30pct_clean.csv", combined)
    (OUT / "manifest.txt").write_text(
        "train_rows=677\nsynthetic_rows=203\ncombined_rows=880\n"
        f"dev_sha256={dev_hash_before}\n"
        f"train_prefixes={Counter(re.match(r'^[A-Za-z]+',r['id']).group() for r in train)}\n"
        f"synthetic_prefixes={Counter(re.match(r'^[A-Za-z]+',r['id']).group() for r in synthetic)}\n",
        encoding="utf-8",
    )
    print("cleaned_train=677 cleaned_synthetic=203 cleaned_combined=880")
    print("train_prefixes", Counter(re.match(r"^[A-Za-z]+", r["id"]).group() for r in train))
    print("synthetic_prefixes", Counter(re.match(r"^[A-Za-z]+", r["id"]).group() for r in synthetic))
    print("dev_sha256_unchanged", dev_hash_before)
    print("output_dir", OUT)


if __name__ == "__main__":
    main()
