"""双模型评测的引用归属和分歧门槛；固定响应，无真实调用。"""

import json

import pytest

from jobcompass import query_policy_review as review


@pytest.mark.parametrize("change", [dict(evidence=["f9"]), dict(grade=True), dict(evidence=[])])
def test_secondary_review_rejects_fabricated_evidence_or_grade(change):
    pair = dict(fragments=[dict(line=1, text="负责Python接口开发")])
    value = dict(grade=2, reason="直接负责接口开发", evidence=["f1"])
    assert review.validate(value, pair) == value
    with pytest.raises(ValueError):
        review.validate(value | change, pair)


def test_merge_requires_adjudication_and_preserves_original_model_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "freeze", lambda directory: None)

    def write(name, items):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")

    primary = [
        dict(pair_id="a", grade=2, source="model", model="Qwen"),
        dict(pair_id="b", grade=0, source="model", model="Qwen"),
    ]
    write(
        "pool_dev.jsonl",
        [dict(pair_id="a", query_id="q", job_key="j1"), dict(pair_id="b", query_id="q", job_key="j2")],
    )
    write("runs.jsonl", [dict(dataset="main", split="dev", query_id="q", ranking=["j1"])])
    write("labels_dev.jsonl", primary)
    write("second_labels_dev.jsonl", [dict(pair_id="a", grade=0, reason="只有相邻方向")])
    original = (tmp_path / "labels_dev.jsonl").read_bytes()
    with pytest.raises(ValueError):
        review.merge(tmp_path, "dev")
    assert (tmp_path / "labels_dev.jsonl").read_bytes() == original
    write(
        "adjudications_dev.jsonl",
        [
            dict(
                pair_id="a",
                grade=1,
                source="assistant_local_JD_adjudication",
                reason="部分职责重合",
                lines=[1],
            )
        ],
    )
    result = review.merge(tmp_path, "dev")
    assert (tmp_path / "labels_dev.model.jsonl").read_bytes() == original
    labels = {x["pair_id"]: x for x in review.rows(tmp_path / "labels_dev.jsonl")}
    assert labels["a"]["grade"] == 1 and labels["b"]["grade"] == 0
    assert result["final_pairs"] == 1


def test_strategy_metrics_require_finished_common_pool_review(tmp_path):
    from jobcompass.query_policy_evaluation import summarize

    with pytest.raises(ValueError, match="merge"):
        summarize(tmp_path, "test")


def test_assistant_first_label_is_not_mislabeled_as_two_model_consensus(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "freeze", lambda directory: None)

    def write(name, items):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")

    write("pool_dev.jsonl", [dict(pair_id="a", query_id="q", job_key="j1")])
    write("runs.jsonl", [dict(dataset="main", split="dev", query_id="q", ranking=["j1"])])
    write(
        "labels_dev.jsonl",
        [
            dict(
                pair_id="a",
                grade=2,
                source="assistant_adjudication_after_two_invalid_judge_responses",
                model=None,
            )
        ],
    )
    write("second_labels_dev.jsonl", [dict(pair_id="a", grade=2, reason="有实际职责")])
    review.merge(tmp_path, "dev")
    assert review.rows(tmp_path / "labels_dev.jsonl")[0]["source"] == "assistant_and_secondary_agreement"
