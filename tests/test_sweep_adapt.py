"""Sweep scoring and eval-subset helpers (no torch required)."""

from dma_kws.stage2.sweep_adapt import (
    compute_sweep_score,
    disable_persistent_workers,
    select_eval_subset_indices,
)


def test_subset_is_spread_over_the_whole_eval_set():
    indices = select_eval_subset_indices(270_684, 2000)

    assert indices is not None
    assert len(indices) == 2000
    assert indices == sorted(indices)
    assert len(set(indices)) == 2000
    # A head slice would never reach the multi-word CSVs appended at the end.
    assert max(indices) > 200_000


def test_subset_is_reproducible_across_calls():
    assert select_eval_subset_indices(10_000, 100) == select_eval_subset_indices(10_000, 100)


def test_subset_disabled_returns_none():
    assert select_eval_subset_indices(1000, 0) is None
    assert select_eval_subset_indices(1000, 5000) is None


def test_trial_config_opts_out_of_persistent_workers():
    """Regression: persistent workers leaked file descriptors across trials."""
    config = {"stage2": {"dataloader": {"persistent_workers": True, "prefetch_factor": 4}}}

    updated = disable_persistent_workers(config)

    assert updated["stage2"]["dataloader"] == {
        "persistent_workers": False,
        "prefetch_factor": 4,
    }
    assert disable_persistent_workers({})["stage2"]["dataloader"] == {"persistent_workers": False}


def test_sweep_score_penalizes_forgetting_only():
    improved = compute_sweep_score(target_auc=0.9, lph_auc_adapted=0.80, lph_auc_base=0.75)
    forgot = compute_sweep_score(target_auc=0.9, lph_auc_adapted=0.70, lph_auc_base=0.75)

    assert improved == 0.9
    assert forgot == 0.85
