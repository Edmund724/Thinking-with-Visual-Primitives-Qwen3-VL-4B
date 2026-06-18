"""Tests for process reward and geometry utilities in metrics module."""

import numpy as np
import pytest

from src.utils.metrics import (
    box_count_answer_consistency_reward,
    box_iou,
    compute_total_reward,
    extract_answer,
    extract_reasoning,
    format_reward,
    is_rollout_correct,
    lenient_parse_boxes,
    match_boxes,
    match_points,
    point_distance,
    primitive_format_compliance_reward,
    process_reward,
    split_generated_text,
    syntax_valid,
)


class TestExtractAnswer:
    def test_boxed_answer(self):
        text = r"Some reasoning \boxed{42}"
        assert extract_answer(text) == "42"

    def test_answer_tag(self):
        text = "Reasoning... <answer>True</answer>"
        assert extract_answer(text) == "True"

    def test_after_think_close(self):
        text = "</think>\n\nThe answer is 7."
        assert extract_answer(text) == "The answer is 7."

    def test_no_answer(self):
        text = "Just some random text without any answer format."
        assert extract_answer(text) is not None  # falls back to cleaned text


class TestExtractReasoning:
    def test_think_tags(self):
        text = "Prefix <think>Step 1. Step 2.</think> answer"
        assert extract_reasoning(text) == "Step 1. Step 2."

    def test_generated_text_no_open_think(self):
        text = "Step 1. Step 2.\n</think>\n\nAnswer"
        assert extract_reasoning(text) == "Step 1. Step 2."

    def test_no_think_tags(self):
        text = "Just plain text."
        assert extract_reasoning(text) == "Just plain text."


class TestSplitGeneratedText:
    def test_split(self):
        reasoning, answer = split_generated_text(
            "<think>I see two cats.</think>\n\nThe answer is 2."
        )
        assert reasoning == "I see two cats."
        assert answer == "The answer is 2."


class TestBoxIOU:
    def test_identical_boxes(self):
        box = (0, 0, 10, 10)
        assert box_iou(box, box) == 1.0

    def test_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert box_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        expected = 25 / 175  # intersect=5*5=25, union=100+100-25=175
        assert box_iou(a, b) == pytest.approx(expected)


class TestMatchBoxes:
    def test_perfect_match(self):
        pred = [(0, 0, 10, 10), (20, 20, 30, 30)]
        gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
        avg_iou, num_match, num_gt = match_boxes(pred, gt, iou_threshold=0.5)
        assert avg_iou == 1.0
        assert num_match == 2
        assert num_gt == 2

    def test_partial_match(self):
        pred = [(0, 0, 10, 10)]
        gt = [(0, 0, 10, 10), (100, 100, 200, 200)]
        avg_iou, num_match, num_gt = match_boxes(pred, gt, iou_threshold=0.5)
        assert num_match == 1
        assert num_gt == 2

    def test_empty_pred(self):
        pred = []
        gt = [(0, 0, 10, 10)]
        avg_iou, num_match, num_gt = match_boxes(pred, gt)
        assert num_match == 0
        assert num_gt == 1
        assert avg_iou == 0.0


class TestPointDistance:
    def test_distance(self):
        assert point_distance((0, 0), (3, 4)) == pytest.approx(5.0)


class TestMatchPoints:
    def test_perfect_match(self):
        pred = [(0, 0), (100, 100)]
        gt = [(0, 0), (100, 100)]
        avg_dist, num_match, num_gt = match_points(pred, gt, dist_threshold=10.0)
        assert avg_dist == 0.0
        assert num_match == 2
        assert num_gt == 2

    def test_no_match_too_far(self):
        pred = [(0, 0)]
        gt = [(1000, 1000)]
        avg_dist, num_match, num_gt = match_points(pred, gt, dist_threshold=10.0)
        assert num_match == 0
        assert num_gt == 1

    def test_empty_pred(self):
        pred = []
        gt = [(0, 0)]
        avg_dist, num_match, num_gt = match_points(pred, gt)
        assert num_match == 0
        assert num_gt == 1
        assert avg_dist == 0.0


class TestSyntaxValid:
    def test_valid(self):
        text = "<|box|>[[1,2,3,4]]<|/box|> <|point|>[[5,6]]<|/point|>"
        assert syntax_valid(text) is True

    def test_unmatched_box(self):
        text = "<|box|>[[1,2,3,4]]"
        assert syntax_valid(text) is False

    def test_unmatched_point(self):
        text = "<|point|>[[5,6]]<|/point|> <|point|>[[7,8]]"
        assert syntax_valid(text) is False


class TestProcessReward:
    def test_box_task_correct(self):
        pred = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        gt = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        r = process_reward(pred, gt, task_type="box")
        assert r["answer_correct"] is True
        assert r["syntax_valid"] is True
        assert r["box_avg_iou"] == 1.0

    def test_box_task_wrong_answer(self):
        pred = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 2."
        gt = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        r = process_reward(pred, gt, task_type="box")
        assert r["answer_correct"] is False

    def test_maze_task_with_collision(self):
        # 3x3 grid, wall at center
        grid = np.ones((3, 3), dtype=np.uint8)
        grid[1, 1] = 0
        pred = (
            "<think><|point|>[[0,0]]<|/point|> <|point|>[[999,999]]<|/point|></think>\n\n"
            "The answer is True."
        )
        gt = (
            "<think><|point|>[[0,0]]<|/point|> <|point|>[[999,999]]<|/point|></think>\n\n"
            "The answer is True."
        )
        r = process_reward(pred, gt, task_type="maze", maze_grid=grid)
        assert r["wall_collision_count"] > 0

    def test_syntax_invalid(self):
        pred = "<|box|>[[1,2,3,4]]"
        gt = "<|box|>[[1,2,3,4]]<|/box|>"
        r = process_reward(pred, gt, task_type="box")
        assert r["syntax_valid"] is False

    def test_point_task(self):
        pred = "<think><|point|>[[100,200]]<|/point|></think>\n\nThe answer is A."
        gt = "<think><|point|>[[100,200]]<|/point|></think>\n\nThe answer is A."
        r = process_reward(pred, gt, task_type="point")
        assert r["answer_correct"] is True
        assert r["point_avg_dist"] == 0.0

    def test_path_task(self):
        """path task_type should be treated like point for reward computation."""
        pred = "<think><|point|>[[100,200]]<|/point|></think>\n\nThe answer is A."
        gt = "<think><|point|>[[100,200]]<|/point|></think>\n\nThe answer is A."
        r = process_reward(pred, gt, task_type="path")
        assert r["answer_correct"] is True
        assert r["point_avg_dist"] == 0.0
        assert "box_avg_iou" in r


class TestPrimitiveFormatComplianceReward:
    def test_valid_ordered_tags(self):
        text = "<|box|>[[1,2,3,4]]<|/box|> <|point|>[[5,6]]<|/point|>"
        assert primitive_format_compliance_reward(text) == pytest.approx(0.2)

    def test_paired_but_wrong_order(self):
        text = "<|/box|>[[1,2,3,4]]<|box|>"
        score = primitive_format_compliance_reward(text)
        assert 0.0 < score < 0.2

    def test_unpaired_box(self):
        text = "<|box|>[[1,2,3,4]]"
        assert primitive_format_compliance_reward(text) < 0.0


class TestBoxCountConsistencyReward:
    def test_count_matches_boxes(self):
        pred = "<think><|box|>[[1,2,3,4],[5,6,7,8]]<|/box|></think>\n\nThe answer is 2."
        gt = "<think><|box|>[[1,2,3,4],[5,6,7,8]]<|/box|></think>\n\nThe answer is 2."
        assert box_count_answer_consistency_reward(pred, gt, "box") == pytest.approx(0.2)

    def test_count_mismatch(self):
        pred = "<think><|box|>[[1,2,3,4]]<|/box|></think>\n\nThe answer is 2."
        gt = "<think><|box|>[[1,2,3,4],[5,6,7,8]]<|/box|></think>\n\nThe answer is 2."
        assert box_count_answer_consistency_reward(pred, gt, "box") == pytest.approx(-0.1)

    def test_negative_sample_no_boxes_expected(self):
        pred = "<think>No objects found.</think>\n\nThe answer is \\boxed{0}."
        gt = "<think>No objects found.</think>\n\nThe answer is \\boxed{0}."
        assert box_count_answer_consistency_reward(pred, gt, "box") == pytest.approx(0.2)

    def test_negative_sample_with_unexpected_boxes(self):
        pred = "<think><|box|>[[1,2,3,4]]<|/box|></think>\n\nThe answer is \\boxed{0}."
        gt = "<think>No objects found.</think>\n\nThe answer is \\boxed{0}."
        assert box_count_answer_consistency_reward(pred, gt, "box") == pytest.approx(-0.1)

    def test_non_box_task(self):
        pred = "<think></think>\n\nThe answer is 2."
        gt = "<think></think>\n\nThe answer is 2."
        assert box_count_answer_consistency_reward(pred, gt, "point") == pytest.approx(0.0)


class TestFormatRewardNoNestedBug:
    def test_valid_box_gets_no_nested_bonus(self):
        text = "<think><|box|>[[1,2,3,4]]<|/box|></think>"
        details = format_reward(text)
        assert details["no_nested_tokens"] is True


class TestNonLatinPenalty:
    def test_non_latin_penalty_heavy(self):
        text = "<think><|box|>[[1,2,3,4]]<|/box|> อันตราย</think>"
        details = format_reward(text)
        assert details["non_latin_penalty"] <= -0.1

    def test_is_rollout_correct_rejects_non_latin(self):
        pred = "<think>อันตราย<|box|>[[1,2,3,4]]<|/box|></think>\n\nThe answer is 1."
        gt = "<think><|box|>[[1,2,3,4]]<|/box|></think>\n\nThe answer is 1."
        assert is_rollout_correct(pred, gt, "box") is False

    def test_lenient_parse_boxes_handles_double_comma(self):
        text = "[[300,581,,334,699]]"
        boxes = lenient_parse_boxes(text)
        assert len(boxes) == 1
        assert boxes[0] == (300, 581, 334, 699)


class TestQualityRewardNonLatin:
    def test_non_latin_major_issue(self):
        pred = "<think><|box|>[[1,2,3,4]]<|/box|> จีน</think>\n\nThe answer is 1."
        gt = "<think><|box|>[[1,2,3,4]]<|/box|></think>\n\nThe answer is 1."
        from src.utils.metrics import quality_reward_text
        assert quality_reward_text(pred, gt, "box") == 0.0


class TestComputeTotalReward:
    def test_perfect_box_includes_new_rewards(self):
        pred = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        gt = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        r = compute_total_reward(pred, gt, task_type="box")
        assert "format_compliance_reward" in r
        assert "box_count_consistency_reward" in r
        assert r["box_count_consistency_reward"] == pytest.approx(0.2)
        assert r["total_reward"] > 1.0

    def test_wrong_tag_order_penalized(self):
        pred = "<think><|box|>[[0,0,10,10]]<|box|></think>\n\nThe answer is 1."
        gt = "<think><|box|>[[0,0,10,10]]<|/box|></think>\n\nThe answer is 1."
        r = compute_total_reward(pred, gt, task_type="box")
        assert r["format_compliance_reward"] < 0.2
        assert r["box_count_consistency_reward"] == pytest.approx(-0.1)
