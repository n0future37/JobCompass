import json
from types import SimpleNamespace

import pytest

from jobcompass.chunking import token_chunks
from jobcompass.experiments import bootstrap_delta, human_metrics
from jobcompass.query_transform import transform_queries, validate_batch


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        result = {
            "input_ids": list(range(len(text) + (0 if kwargs.get("add_special_tokens") is False else 2)))
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(n, n + 1) for n in range(len(text))]
        return result


def test_token_chunk_budget_covers_long_line_and_keeps_original_offsets(job_factory):
    job = job_factory(jd="Python" * 100 + "\nMySQL接口\n" + "后端" * 100)
    tokenizer = CharacterTokenizer()
    passages = token_chunks([job], tokenizer, budget=64, overlap=8)
    assert len(passages) > 1
    seen = set()
    for row in passages:
        assert row["text"] == job["title"] + "\n" + job["jd_text"][row["char_start"] : row["char_end"]]
        assert len(tokenizer(row["text"])["input_ids"]) <= 64
        seen.update(range(row["char_start"], row["char_end"]))
        assert row["line_start"] <= row["line_end"]
    assert len(seen) == len(job["jd_text"])


def test_partial_human_labels_produce_bounds_not_fake_precision():
    result = human_metrics(["a", "b", "c"], {"a": 2, "b": 0})
    assert result["precision_at_10"] is None and result["pooled_ndcg_at_10"] is None
    assert result["precision_lower_bound"] == 0.1 and result["precision_upper_bound"] == 0.2
    assert result["unjudged_keys"] == ["c"]


def test_transform_rejects_missing_duplicate_or_fabricated_ids():
    queries = [{"query_id": "a"}]
    with pytest.raises(ValueError):
        validate_batch({"items": [{"query_id": "b", "text": "Python接口"}]}, queries, "rewrite")
    with pytest.raises(ValueError):
        validate_batch({"items": []}, queries, "hyde")
    assert validate_batch({"items": [{"query_id": "a", "text": "Python接口"}]}, queries, "rewrite") == {
        "a": "Python接口"
    }


def test_transform_payload_excludes_profile_and_resume_uses_cache(tmp_path, monkeypatch):
    from jobcompass import query_transform

    captured = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            captured.append(json)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "output_text": '{"items":[{"query_id":"a","text":"Python接口与服务开发"}]}',
                    "usage": {"total_tokens": 10},
                },
            )

    monkeypatch.setattr(query_transform.httpx, "Client", Client)
    monkeypatch.setenv("UNIT_TRANSFORM_KEY", "fake-unit-key")
    settings = SimpleNamespace(
        api_config={
            "API_BASE": "https://unit.invalid/v1",
            "API_KEY_ENV": "UNIT_TRANSFORM_KEY",
            "MODEL": "fixed",
            "REASONING_EFFORT": "",
        }
    )
    queries = [
        {"query_id": "a", "query": "Python接口", "kind": "human", "profile": "PRIVATE_PROFILE_SENTINEL"}
    ]
    result = transform_queries(settings, queries, tmp_path)
    assert result["reported_total_tokens"] == 20 and len(captured) == 2
    assert "PRIVATE_PROFILE_SENTINEL" not in json.dumps(captured)
    assert transform_queries(settings, queries, tmp_path) == result and len(captured) == 2


def test_bootstrap_clusters_two_queries_of_same_anchor():
    rows = []
    for variant, hit in (("a", 0), ("b", 1)):
        for q in ("q1", "q2"):
            rows.append(
                {
                    "query_id": q,
                    "variant": variant,
                    "kind": "anchor",
                    "split": "test",
                    "group": "same_anchor",
                    "anchor_hit_at_10": hit,
                }
            )
    result = bootstrap_delta(rows, "a", "b")
    assert result["groups"] == 1 and result["delta"] == 1 and result["interval_95"] == [1, 1]


def test_hyde_fallback_restores_query_prefix():
    from jobcompass.experiments import dense_query_prefix

    assert dense_query_prefix("hyde", "D01", "human", {"D01"})
    assert not dense_query_prefix("hyde", "D02", "human", {"D01"})
    assert dense_query_prefix("hyde", "C01", "control", set())


def test_explicit_corpus_snapshot_rejects_drift_or_tampering(tmp_path, job_factory):
    from jobcompass.experiments import verify_corpus_snapshot
    from jobcompass.preprocessing import digest

    job = job_factory()
    body = (json.dumps(job, ensure_ascii=False) + "\n").encode()
    (tmp_path / "jobs.jsonl").write_bytes(body)
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(jobs=1, jobs_sha256=digest(body))), encoding="utf-8"
    )
    assert verify_corpus_snapshot([job], tmp_path)["jobs"] == 1
    with pytest.raises(ValueError, match="岗位/版本"):
        verify_corpus_snapshot([job | {"version": 2}], tmp_path)
    with pytest.raises(ValueError, match="岗位/版本"):
        verify_corpus_snapshot([], tmp_path)
    (tmp_path / "jobs.jsonl").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="指纹"):
        verify_corpus_snapshot([job], tmp_path)


def test_new_corpus_cannot_use_implicit_old_human_queries(tmp_path):
    from jobcompass.experiments import run

    with pytest.raises(ValueError, match="不能混用旧人标"):
        run(directory=tmp_path / "out", corpus_snapshot=tmp_path)
    assert not (tmp_path / "out").exists()
