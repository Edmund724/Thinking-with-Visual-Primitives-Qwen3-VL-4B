"""Format Reward Model (Format RM) for visual primitives."""

import re
from typing import Dict

from ..constants import BOX_OPEN, BOX_CLOSE, POINT_OPEN, POINT_CLOSE, BOX_PATTERN, POINT_PATTERN, REF_OPEN, REF_CLOSE, REF_PATTERN


# Regex matching characters from non-Latin scripts (Cyrillic, Arabic, CJK, Thai, etc.)
_NON_LATIN_SCRIPT_RE = re.compile(
    r"["
    r"\u0370-\u03FF\u1F00-\u1FFF"  # Greek
    r"\u0400-\u04FF"               # Cyrillic
    r"\u0600-\u06FF"               # Arabic
    r"\u0900-\u097F"               # Devanagari
    r"\u0E00-\u0E7F"               # Thai
    r"\u3040-\u309F\u30A0-\u30FF"  # Hiragana/Katakana
    r"\u4E00-\u9FFF"               # CJK Unified Ideographs
    r"\uAC00-\uD7AF"               # Korean Hangul Syllables
    r"]"
)


def format_reward(text: str) -> Dict:
    """Format Reward Model (Format RM).

    Checks:
        1. <think>...</think> tags exist and are paired
        2. Special token syntax is valid (paired open/close)
        3. Coordinates are within [0, 999]
        4. No duplicate special tokens (e.g., nested boxes)
        5. Output is in English (no non-Latin script characters)

    Returns:
        Dict with individual check results and total score (0.0~0.8, can go
        negative due to the non-Latin penalty).
    """
    score = 0.0
    details = {}

    # 1. Check <think> tags
    think_open = text.count("<think>")
    think_close = text.count("</think>")
    # Qwen3-VL chat template prepends <think> to prompt;
    # GRPO completions only have </think> — accept that too.
    has_think = (think_open == 1 and think_close == 1) or (think_close >= 1 and think_open == 0)
    details["has_think_tags"] = has_think
    if has_think:
        score += 0.2

    # 2. Check special token pairing
    box_open = text.count(BOX_OPEN)
    box_close = text.count(BOX_CLOSE)
    point_open = text.count(POINT_OPEN)
    point_close = text.count(POINT_CLOSE)
    tokens_paired = (
        box_open == box_close and point_open == point_close
        and box_open >= 0 and point_open >= 0
    )
    details["tokens_paired"] = tokens_paired
    if tokens_paired:
        score += 0.2

    # 3. Check coordinate range [0, 999]
    coords_valid = True
    for match in BOX_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            nums = [int(x) for x in re.findall(r"\d+", coords_str)]
            for n in nums:
                if n < 0 or n > 999:
                    coords_valid = False
                    break
        except (ValueError, IndexError):
            coords_valid = False
        if not coords_valid:
            break

    for match in POINT_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            nums = [int(x) for x in re.findall(r"\d+", coords_str)]
            for n in nums:
                if n < 0 or n > 999:
                    coords_valid = False
                    break
        except (ValueError, IndexError):
            coords_valid = False
        if not coords_valid:
            break

    details["coords_in_range"] = coords_valid
    if coords_valid:
        score += 0.2

    # 4. Check no duplicate/nested special tokens and correct inner bracket format.
    # Correct: <|box|>[[x1,y1,x2,y2]]<|/box|> or
    #          <|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>
    # Wrong:   <|box|>[[[x1,...],[x2,...]]]<|/box|> (extra bracket around each box)
    no_duplicates = True
    box_segments = re.findall(
        re.escape(BOX_OPEN) + r"(.*?)" + re.escape(BOX_CLOSE), text, re.DOTALL
    )
    for seg in box_segments:
        # Nested / duplicate tags inside a segment are wrong; the inner [[..]]
        # bracket wrapper is part of the valid format and must not be flagged.
        if BOX_OPEN in seg or BOX_CLOSE in seg:
            no_duplicates = False
            break

    point_segments = re.findall(
        re.escape(POINT_OPEN) + r"(.*?)" + re.escape(POINT_CLOSE), text, re.DOTALL
    )
    for seg in point_segments:
        if POINT_OPEN in seg or POINT_CLOSE in seg:
            no_duplicates = False
            break

    details["no_nested_tokens"] = no_duplicates
    if no_duplicates:
        score += 0.2

    # 5. Penalize non-Latin script characters (Thai, Cyrillic, CJK, etc.)
    non_latin_chars = len(_NON_LATIN_SCRIPT_RE.findall(text))
    details["non_latin_char_count"] = non_latin_chars
    if non_latin_chars > 0:
        # Heavy penalty: even one non-Latin character is a strong signal that
        # the model is drifting out of the desired output distribution.
        penalty = min(0.1 * non_latin_chars, 1.0)
        score -= penalty
        details["non_latin_penalty"] = round(-penalty, 2)
    else:
        details["non_latin_penalty"] = 0.0

    # 6. Check ref-box pairing: every <|box|> should be preceded by a <|ref|>
    ref_pairing = _check_ref_box_pairing(text)
    details["ref_box_pairing"] = ref_pairing
    score += ref_pairing["score"]

    details["total_format_score"] = round(score, 2)
    return details


def _check_ref_box_pairing(text: str) -> Dict:
    """Check whether <|box|> tags are preceded by <|ref|> tags.

    Returns dict with:
        - score: float in [-0.15, 0.0]
        - num_boxes: int
        - num_refs: int
        - unpaired_boxes: int
    """
    box_positions = [m.start() for m in BOX_PATTERN.finditer(text)]
    ref_positions = [m.start() for m in REF_PATTERN.finditer(text)]

    num_boxes = len(box_positions)
    num_refs = len(ref_positions)

    if num_boxes == 0:
        # No boxes — no pairing requirement
        return {"score": 0.0, "num_boxes": 0, "num_refs": num_refs, "unpaired_boxes": 0}

    # Count boxes without a preceding ref
    unpaired = 0
    for bp in box_positions:
        has_ref_before = any(rp < bp for rp in ref_positions)
        if not has_ref_before:
            unpaired += 1

    # Also check ref content quality
    bad_refs = 0
    for m in REF_PATTERN.finditer(text):
        content = m.group(1).strip()
        if not content or content.isdigit():
            bad_refs += 1

    # Scoring
    score = 0.0
    if unpaired > 0:
        score -= min(0.1 * unpaired, 0.1)  # max -0.1 for unpaired boxes
    if bad_refs > 0:
        score -= min(0.05 * bad_refs, 0.05)  # max -0.05 for bad ref content

    return {
        "score": round(score, 2),
        "num_boxes": num_boxes,
        "num_refs": num_refs,
        "unpaired_boxes": unpaired,
        "bad_refs": bad_refs,
    }


def primitive_format_compliance_reward(text: str) -> float:
    """Reward for correctly ordered and paired primitive tags.

    Unlike ``format_reward`` which only checks equal open/close counts,
    this reward explicitly checks that tags appear in the right order
    (open ... close) and penalizes malformed / unpaired tags.

    Returns:
        Score in [-0.2, 0.2].
    """
    box_open = text.count(BOX_OPEN)
    box_close = text.count(BOX_CLOSE)
    point_open = text.count(POINT_OPEN)
    point_close = text.count(POINT_CLOSE)

    score = 0.0

    # 1) Paired counts
    if box_open == box_close and point_open == point_close:
        score += 0.1

    # 2) Correct order: each open is followed by its close.
    box_segments = re.findall(
        re.escape(BOX_OPEN) + r"(.*?)" + re.escape(BOX_CLOSE), text, re.DOTALL
    )
    point_segments = re.findall(
        re.escape(POINT_OPEN) + r"(.*?)" + re.escape(POINT_CLOSE), text, re.DOTALL
    )
    ordered_ok = len(box_segments) == box_open and len(point_segments) == point_open
    if ordered_ok:
        score += 0.1

    # 3) Penalize malformed / unpaired tags
    malformed = (
        abs(box_open - box_close)
        + abs(point_open - point_close)
    )
    if not ordered_ok:
        # Additional penalty when open/close counts match but order is wrong
        malformed += 1
    score -= 0.05 * malformed

    return float(max(min(score, 0.2), -0.2))
