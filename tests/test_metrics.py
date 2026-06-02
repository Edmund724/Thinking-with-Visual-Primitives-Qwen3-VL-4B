"""Tests for process reward and geometry utilities in metrics module."""

import numpy as np
import pytest

from src.utils.metrics import (
    box_iou,
    extract_answer,
    extract_reasoning,
    match_boxes,
    match_points,
    point_distance,
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
