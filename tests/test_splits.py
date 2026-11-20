import pytest

from roadshift.data.splits import (
    make_blocks,
    natural_key,
    nested_fewshot,
    shuffled_blocks,
    split_blocks,
    subsample_backgrounds,
    take_blocks,
)

STEMS = [f"Japan_{i:06d}" for i in range(1, 101)]


def test_natural_key_orders_numbers():
    assert sorted(["a_10", "a_2", "a_1"], key=natural_key) == ["a_1", "a_2", "a_10"]


def test_blocks_are_contiguous_and_complete():
    blocks = make_blocks(list(reversed(STEMS)), 7)
    assert [s for b in blocks for s in b] == STEMS
    assert all(len(b) == 7 for b in blocks[:-1]) and len(blocks[-1]) == 100 % 7


def test_shuffle_is_seeded():
    assert shuffled_blocks(STEMS, 5, 1) == shuffled_blocks(STEMS, 5, 1)
    assert shuffled_blocks(STEMS, 5, 1) != shuffled_blocks(STEMS, 5, 2)


def test_take_blocks_exact_count():
    kept = take_blocks(shuffled_blocks(STEMS, 7, 0), 30)
    assert sum(map(len, kept)) == 30
    assert take_blocks(make_blocks(STEMS, 7), None) == make_blocks(STEMS, 7)


def test_split_fractions_disjoint_complete_and_block_preserving():
    blocks = shuffled_blocks(STEMS, 5, 0)
    out = split_blocks(blocks, {"train": 0.7, "val": 0.15, "test": 0.15})
    all_ = out["train"] + out["val"] + out["test"]
    assert sorted(all_) == sorted(STEMS) and len(set(all_)) == len(all_)
    assert len(out["train"]) == 70 and len(out["val"]) == 15 and len(out["test"]) == 15
    where = {s: k for k, v in out.items() for s in v}
    assert all(len({where[s] for s in b}) == 1 for b in blocks)  # aucun bloc coupé


def test_split_never_overshoots_by_more_than_half_a_block():
    # Régression : le remplissage glouton mettait 60/60 images en train avec 3 blocs de 20.
    blocks = make_blocks(STEMS[:60], 20)
    out = split_blocks(blocks, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert [len(out[k]) for k in ("train", "val", "test")] == [
        40,
        20,
        0,
    ]  # train vise 30 -> 40 (et non 60) ; val vise 15 -> 20
    out = split_blocks(shuffled_blocks(STEMS, 3, 0), {"train": 0.7, "val": 0.15, "test": 0.15})
    assert abs(len(out["train"]) - 70) <= 1.5 and abs(len(out["val"]) - 15) <= 1.5


def test_split_absolute_and_rest():
    out = split_blocks(shuffled_blocks(STEMS, 10, 0), {"test": 40, "pool": "rest"})
    assert len(out["test"]) == 40 and len(out["pool"]) == 60


@pytest.mark.parametrize(
    "spec",
    [{"a": "rest", "b": 0.5}, {"a": 1.5}, {"a": 0}, {"a": 200}, {"a": True}, {}],
)
def test_split_invalid_specs(spec):
    with pytest.raises(ValueError):
        split_blocks(make_blocks(STEMS, 10), spec)


def test_fewshot_nested_and_seeded():
    sub = nested_fewshot(STEMS, [50, 10, 20], seed=3)
    assert set(sub[10]) <= set(sub[20]) <= set(sub[50])
    assert [len(sub[k]) for k in (10, 20, 50)] == [10, 20, 50]
    assert sub == nested_fewshot(list(reversed(STEMS)), [10, 20, 50], seed=3)  # indépendant de l'ordre
    assert sub[10] != nested_fewshot(STEMS, [10], seed=4)[10]
    with pytest.raises(ValueError):
        nested_fewshot(STEMS[:5], [10], seed=0)


def test_subsample_backgrounds():
    bg = {s: i % 2 == 0 for i, s in enumerate(STEMS)}
    assert subsample_backgrounds(STEMS, bg, 1.0, 0) == STEMS
    assert subsample_backgrounds(STEMS, bg, 0.0, 0) == [s for s in STEMS if not bg[s]]
    half = subsample_backgrounds(STEMS, bg, 0.5, 0)
    assert half == subsample_backgrounds(STEMS, bg, 0.5, 0)
    with pytest.raises(ValueError):
        subsample_backgrounds(STEMS, bg, 1.5, 0)
