"""Panel tables: sharded scoring of one ladder is joined back into one table."""

import json

import pytest

from geosae.eval.ladder import merge_panel_shards, write_panel_table


def _write_shard(tmp_path, name, rows, provenance=None):
    d = tmp_path / name
    d.mkdir()
    path = d / "panel.json"
    path.write_text(json.dumps({"_provenance": provenance or {"store": "s"}, "rows": rows}))
    return path


def test_merge_panel_shards_concatenates_disjoint_rows(tmp_path):
    a = _write_shard(tmp_path, "shard_a", [{"name": "r1", "fve": 0.1}, {"name": "r2", "fve": 0.2}])
    b = _write_shard(tmp_path, "shard_b", [{"name": "r3", "fve": 0.3}])
    rows, provenance = merge_panel_shards([a, b])
    assert {r["name"] for r in rows} == {"r1", "r2", "r3"}
    assert provenance["shards"] == [str(a), str(b)]
    assert provenance["n_shards"] == 2
    assert provenance["store"] == "s"  # kept from the first shard


def test_merge_panel_shards_lets_a_later_shard_win_a_name_collision(tmp_path):
    a = _write_shard(tmp_path, "shard_a", [{"name": "r1", "fve": 0.1}])
    b = _write_shard(tmp_path, "shard_b", [{"name": "r1", "fve": 0.9}])
    rows, _ = merge_panel_shards([a, b])
    assert len(rows) == 1
    assert rows[0]["fve"] == 0.9


def test_merge_panel_shards_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no shards"):
        merge_panel_shards([])


def test_write_panel_table_round_trips_through_merge(tmp_path):
    rows = [{"name": "r1", "fve": 0.5}, {"name": "r2", "fve": 0.6}]
    write_panel_table(tmp_path / "eval", rows, {"store": "s"})
    assert (tmp_path / "eval" / "panel.csv").exists()
    merged, provenance = merge_panel_shards([tmp_path / "eval" / "panel.json"])
    assert merged == rows and provenance["store"] == "s"
