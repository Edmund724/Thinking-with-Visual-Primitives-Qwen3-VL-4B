"""Tests for visual primitive formatting and cleaning helpers."""

from src.data.formatters.primitive_formatter import (
    clean_primitive_tags,
    format_box,
    format_point,
)


def test_format_box():
    assert format_box([(1, 2, 3, 4)]) == "<|box|>[[1,2,3,4]]<|/box|>"
    assert (
        format_box([(1, 2, 3, 4), (5, 6, 7, 8)])
        == "<|box|>[[1,2,3,4],[5,6,7,8]]<|/box|>"
    )


def test_format_point():
    assert format_point([(1, 2)]) == "<|point|>[[1,2]]<|/point|>"


def test_clean_reversed_box_tags():
    text = "the cat is at <|/box|>[[10,20,30,40]]<|box|> here"
    fixed = clean_primitive_tags(text, "box")
    assert fixed == "the cat is at <|box|>[[10,20,30,40]]<|/box|> here"


def test_clean_duplicate_open_box_tags():
    text = "the cat is at <|box|>[[10,20,30,40]]<|box|> here"
    fixed = clean_primitive_tags(text, "box")
    assert fixed == "the cat is at <|box|>[[10,20,30,40]]<|/box|> here"


def test_clean_point_tags():
    text = "target at <|/point|>[[100,200]]<|point|>"
    fixed = clean_primitive_tags(text, "point")
    assert fixed == "target at <|point|>[[100,200]]<|/point|>"


def test_clean_bad_token_variants():
    text = "x at <|box_start|>[[1,2,3,4]]<|box_end|>"
    fixed = clean_primitive_tags(text, "box")
    assert fixed == "x at <|box|>[[1,2,3,4]]<|/box|>"
