"""Tests for visual primitive parser."""

import numpy as np
import pytest

from src.models.visual_primitive_parser import PrimitiveParser


class TestPrimitiveParser:
    """Test the PrimitiveParser class."""

    def test_extract_boxes_single(self):
        text = "Found object at <|box|>[[100, 200, 300, 400]]<|/box|>"
        boxes = PrimitiveParser.extract_boxes(text)
        assert len(boxes) == 1
        assert boxes[0] == (100, 200, 300, 400)

    def test_extract_boxes_multiple(self):
        text = "Objects: <|box|>[[100,200,300,400],[500,600,700,800]]<|/box|>"
        boxes = PrimitiveParser.extract_boxes(text)
        assert len(boxes) == 2
        assert boxes[0] == (100, 200, 300, 400)
        assert boxes[1] == (500, 600, 700, 800)

    def test_extract_boxes_none(self):
        text = "No boxes here."
        boxes = PrimitiveParser.extract_boxes(text)
        assert boxes == []

    def test_extract_points_single(self):
        text = "Waypoint: <|point|>[[400, 300]]<|/point|>"
        points = PrimitiveParser.extract_points(text)
        assert len(points) == 1
        assert points[0] == (400, 300)

    def test_extract_points_multiple(self):
        text = "Path: <|point|>[[100,200],[300,400]]<|/point|>"
        points = PrimitiveParser.extract_points(text)
        assert len(points) == 2
        assert points[0] == (100, 200)
        assert points[1] == (300, 400)

    def test_validate_syntax_valid(self):
        text = "Box: <|box|>[[1,2,3,4]]<|/box|> Point: <|point|>[[5,6]]<|/point|>"
        assert PrimitiveParser.validate_syntax(text) is True

    def test_validate_syntax_unmatched(self):
        text = "Box: <|box|>[[1,2,3,4]]"
        assert PrimitiveParser.validate_syntax(text) is False

    def test_validate_syntax_nested_unmatched(self):
        text = "Bad: <|box|>[[1,2,3,4]]<|/box|> <|point|>[[5,6]]"
        assert PrimitiveParser.validate_syntax(text) is False

    def test_validate_coordinates_valid(self):
        text = "<|box|>[[100,200,300,400]]<|/box|> <|point|>[[50,60]]<|/point|>"
        errors = PrimitiveParser.validate_coordinates(text, (1000, 1000))
        assert errors == []

    def test_validate_coordinates_out_of_bounds(self):
        text = "<|box|>[[-1,200,300,1000]]<|/box|>"
        errors = PrimitiveParser.validate_coordinates(text, (1000, 1000))
        assert len(errors) >= 1
        assert "out of bounds" in errors[0]

    def test_check_wall_collision_no_collision(self):
        # Simple 3x3 grid, all path cells (1)
        grid = np.ones((3, 3), dtype=np.uint8)
        text = "<|point|>[[0,0]]<|/point|> <|point|>[[2,2]]<|/point|>"
        # In normalized 0-999, (0,0) -> grid (0,0), (2*332,2*332) -> grid (0,0) or (1,1)
        # This is tricky due to normalization; let's use midpoint
        text = "<|point|>[[0,0]]<|/point|> <|point|>[[0,1]]<|/point|>"
        collisions = PrimitiveParser.check_wall_collision(text, grid)
        assert len(collisions) == 0

    def test_check_wall_collision_with_collision(self):
        # Grid with wall at center
        grid = np.ones((3, 3), dtype=np.uint8)
        grid[1, 1] = 0  # Wall at center
        # Points at (0,0) and (2,2) -> line passes through center -> collision
        text = "<|point|>[[0,0]]<|/point|> <|point|>[[999,999]]<|/point|>"
        collisions = PrimitiveParser.check_wall_collision(text, grid)
        assert len(collisions) > 0

    def test_count_tags(self):
        text = "<|box|>[[1,2,3,4]]<|/box|> <|point|>[[5,6]]<|/point|> <|box|>[[7,8,9,10]]<|/box|>"
        counts = PrimitiveParser.count_tags(text)
        assert counts["box_open"] == 2
        assert counts["box_close"] == 2
        assert counts["point_open"] == 1
        assert counts["point_close"] == 1

    def test_has_backtracking_keywords(self):
        text = "Dead end at <|point|>[[100,200]]<|/point|>, backtracking..."
        assert PrimitiveParser.has_backtracking_keywords(text) is True

    def test_no_backtracking_keywords(self):
        text = "Move to <|point|>[[100,200]]<|/point|>, then to <|point|>[[300,400]]<|/point|>"
        assert PrimitiveParser.has_backtracking_keywords(text) is False
