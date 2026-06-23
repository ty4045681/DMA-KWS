from dma_kws.jsonl import read_jsonl


def test_read_jsonl_parses_lines_and_skips_blanks(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"a": 1}\n'
        "\n"
        '   \n'
        '{"b": "two"}\n',
        encoding="utf-8",
    )

    records = read_jsonl(path)

    assert records == [{"a": 1}, {"b": "two"}]


def test_read_jsonl_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_jsonl(path) == []
