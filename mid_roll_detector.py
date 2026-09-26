"""
Phát hiện half-digit (mid-roll) từ chính output softmax của digit classifier
đã có — KHÔNG cần train model/class riêng cho việc này.

QUY TẮC:
1. Top-2 dự đoán phải là 2 số LIỀN KỀ theo thứ tự con lăn quay
   (0-1, 1-2, ..., 8-9, 9-0 — vòng tròn, không phải liền kề theo hình dạng).
2. Margin giữa top-1 và top-2 phải đủ nhỏ (mặc định < 0.35) mới coi là mid-roll
   thật, tránh nhầm với ca model chỉ hơi không chắc nhưng vẫn có 1 đáp án rõ.
3. Cặp (8,9) và (9,0) dùng ngưỡng margin CHẶT HƠN hẳn (mặc định < 0.15) —
   vì đây cũng là 2 cặp dễ nhầm lẫn hình dạng thông thường (đã thấy trong
   confusion matrix thật: 8->9, 9->0), false positive ở 2 cặp này nguy hiểm
   hơn hẳn (tự tin trả nhầm số thay vì route review), nên thà bỏ sót còn hơn
   nhận nhầm.
4. Giá trị cuối cùng lấy SỐ LỚN HƠN trong 2 lựa chọn (theo quy ước nghiệp vụ
   đã xác nhận: tránh thất thoát điện đã tiêu thụ nhưng công tơ chưa "chốt"
   xong vòng quay).
"""

from dataclasses import dataclass
from typing import Optional

RISKY_PAIRS = {frozenset({"8", "9"}), frozenset({"9", "0"})}


@dataclass
class MidRollResult:
    is_mid_roll: bool
    resolved_value: Optional[str]   # số cuối cùng nên dùng, None nếu không áp dụng
    top1_label: str
    top1_prob: float
    top2_label: str
    top2_prob: float
    margin: float
    reason: str


def is_adjacent(a: str, b: str) -> bool:
    """Liền kề theo thứ tự con lăn quay, có vòng (9 kề 0)."""
    ai, bi = int(a), int(b)
    diff = abs(ai - bi)
    return diff == 1 or diff == 9  # diff=9 tức là cặp (0,9)


def detect_mid_roll(probs: dict, margin_threshold: float = 0.35,
                    risky_margin_threshold: float = 0.15) -> MidRollResult:
    """
    probs: dict {"0": 0.02, "1": 0.60, ..., "9": 0.01} — softmax output của
           digit classifier cho 1 ảnh, đã convert sang dict label->prob.
    """
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top1_label, top1_prob), (top2_label, top2_prob) = ranked[0], ranked[1]
    margin = top1_prob - top2_prob

    if not is_adjacent(top1_label, top2_label):
        return MidRollResult(
            is_mid_roll=False, resolved_value=top1_label,
            top1_label=top1_label, top1_prob=top1_prob,
            top2_label=top2_label, top2_prob=top2_prob,
            margin=margin,
            reason="top-2 không liền kề -> không phải mid-roll, nhận top-1 bình thường",
        )

    pair_key = frozenset({top1_label, top2_label})
    threshold = risky_margin_threshold if pair_key in RISKY_PAIRS else margin_threshold

    if margin >= threshold:
        return MidRollResult(
            is_mid_roll=False, resolved_value=top1_label,
            top1_label=top1_label, top1_prob=top1_prob,
            top2_label=top2_label, top2_prob=top2_prob,
            margin=margin,
            reason=f"margin {margin:.2f} >= ngưỡng {threshold:.2f} -> đủ tự tin, "
                   f"không coi là mid-roll dù liền kề",
        )

    pair = {top1_label, top2_label}
    if pair == {"9", "0"}:
        # Trường hợp đặc biệt: con lăn quay từ 9 sang 0 nghĩa là đã LĂN QUA
        # một vòng — giá trị "tiến xa hơn" là 0 (đã vòng), không phải 9.
        # max(key=int) thông thường sẽ luôn chọn nhầm "9" ở đúng cặp này.
        larger = "0"
    else:
        larger = max(top1_label, top2_label, key=int)
    return MidRollResult(
        is_mid_roll=True, resolved_value=larger,
        top1_label=top1_label, top1_prob=top1_prob,
        top2_label=top2_label, top2_prob=top2_prob,
        margin=margin,
        reason=f"liền kề ({top1_label},{top2_label}), margin {margin:.2f} < "
               f"ngưỡng {threshold:.2f} -> mid-roll, lấy số lớn hơn = {larger}"
               + (" [CẶP RỦI RO, dùng ngưỡng chặt]" if pair_key in RISKY_PAIRS else ""),
    )
