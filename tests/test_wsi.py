from histopathology_pipeline.wsi import discover_slides


def test_discover_slides_is_recursive_and_case_insensitive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.ndpi").write_bytes(b"")
    (nested / "b.SVS").write_bytes(b"")
    (nested / "notes.txt").write_text("ignore", encoding="utf-8")

    slides = discover_slides(tmp_path)

    assert slides == [
        (tmp_path / "a.ndpi").resolve(),
        (nested / "b.SVS").resolve(),
    ]
