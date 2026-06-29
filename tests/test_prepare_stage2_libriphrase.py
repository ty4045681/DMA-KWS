"""Ensure the legacy script name delegates to the paper prep entry point."""

from scripts import prepare_stage2_libriphrase as prep


def test_libriphrase_script_delegates_to_paper_main(monkeypatch):
    called = []

    def fake_main():
        called.append(True)

    monkeypatch.setattr(prep, "main", fake_main)
    prep.main()
    assert called == [True]
