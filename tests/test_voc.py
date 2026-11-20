import pytest

from roadshift.data.voc import (
    VocAnnotation,
    VocObject,
    annotation_to_yolo_lines,
    parse_voc,
    voc_box_to_yolo,
)

from .conftest import voc_xml

CLASSES = ["D00", "D10", "D20", "D40"]


def _ann(objs, w=100, h=80):
    return VocAnnotation("x.jpg", w, h, [VocObject(*o) for o in objs])


def test_parse_voc_reads_size_and_objects(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(voc_xml("a.jpg", 640, 480, [("D00", 1, 2, 30.5, 40)]))
    ann = parse_voc(p)
    assert (ann.filename, ann.width, ann.height) == ("a.jpg", 640, 480)
    assert ann.objects == [VocObject("D00", 1.0, 2.0, 30.5, 40.0)]


def test_parse_voc_missing_size_gives_zero(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(voc_xml("a.jpg", 640, 480, [("D00", 1, 2, 3, 4)], write_size=False))
    assert parse_voc(p).width == 0


def test_parse_voc_skips_object_without_box(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text("<annotation><size><width>10</width><height>10</height></size>"
                 "<object><name>D00</name></object>"
                 "<object><name>D10</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax></bndbox></object>"
                 "</annotation>")  # fmt: skip
    assert parse_voc(p).objects == []


def test_voc_box_to_yolo_math():
    assert voc_box_to_yolo(0, 0, 100, 80, 100, 80) == (0.5, 0.5, 1.0, 1.0)
    cx, cy, w, h = voc_box_to_yolo(10, 20, 30, 60, 100, 80)
    assert (cx, cy, w, h) == pytest.approx((0.2, 0.5, 0.2, 0.5))


def test_lines_format_and_class_ids():
    lines, stats = annotation_to_yolo_lines(_ann([("D40", 10, 20, 30, 60), ("D00", 0, 0, 100, 80)]), CLASSES)
    assert lines == ["3 0.200000 0.500000 0.200000 0.500000", "0 0.500000 0.500000 1.000000 1.000000"]
    assert stats.kept == {"D40": 1, "D00": 1}


def test_unknown_labels_counted_and_aliases_applied():
    ann = _ann([("D43", 1, 1, 20, 20), ("D0w0", 1, 1, 20, 20)])
    lines, stats = annotation_to_yolo_lines(ann, CLASSES, aliases={"D0w0": "D00"})
    assert len(lines) == 1 and lines[0].startswith("0 ")
    assert stats.unknown_labels == {"D43": 1}


def test_boxes_are_clipped_to_image():
    lines, _ = annotation_to_yolo_lines(_ann([("D10", -10, -10, 50, 40)]), CLASSES)
    cls, cx, cy, w, h = lines[0].split()
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.25, 0.25, 0.5, 0.5))
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))


@pytest.mark.parametrize(
    "box,reason",
    [
        (("D00", 50, 10, 40, 30), "inverted_or_empty"),
        (("D00", 10, 10, 10, 30), "inverted_or_empty"),
        (("D00", 120, 10, 150, 30), "outside_image"),
        (("D00", 10, 10, 11, 30), "too_small"),
    ],
)
def test_invalid_boxes_dropped_with_reason(box, reason):
    lines, stats = annotation_to_yolo_lines(_ann([box]), CLASSES, min_box_px=2)
    assert lines == [] and stats.dropped == {reason: 1}


def test_exact_duplicates_removed():
    lines, stats = annotation_to_yolo_lines(_ann([("D00", 1, 1, 20, 20)] * 2), CLASSES)
    assert len(lines) == 1 and stats.dropped == {"duplicate": 1}


def test_size_override_and_invalid_size():
    ann = _ann([("D00", 0, 0, 50, 40)], w=0, h=0)
    with pytest.raises(ValueError):
        annotation_to_yolo_lines(ann, CLASSES)
    lines, _ = annotation_to_yolo_lines(ann, CLASSES, width=100, height=80)
    assert lines == ["0 0.250000 0.250000 0.500000 0.500000"]
