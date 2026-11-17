"""离线核验标注/查询冻结产物；不连接数据库或调用模型。"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, annotations, queries):
    manifest = read_json(annotations / "manifest.json")
    for name, expected in manifest["sources"].items():
        require(checksum(root / name) == expected, f"源文件变化：{name}")
    for name, expected in manifest["files"].items():
        require(checksum(annotations / name) == expected, f"标注产物变化：{name}")
    rows = read_jsonl(annotations / "annotations.jsonl")
    pairs = {(r["query_id"], r["job_key"]) for r in rows}
    require(len(pairs) == len(rows) == manifest["pairs"], "标注缺失或重复")
    require(len({r["id"] for r in rows}) == len(rows), "标注ID重复")
    evidence_count = 0
    for row in rows:
        require(row["annotation_kind"] == "assistant", "本补充包必须如实标为assistant")
        require(row["label"] in (0, 1, 2) and type(row["label"]) is int, "相关性等级不合法")
        require(row["reason"] and row["annotator"] and row["reviewed_at"], "标注元数据缺失")
        require(row["evidence"], "缺少JD行证据")
        lines = row["jd_text"].split("\n")
        for evidence in row["evidence"]:
            number = evidence["line"]
            require(1 <= number <= len(lines), "证据行号越界")
            require(lines[number - 1] == evidence["text"], "证据不是对应JD原文")
            evidence_count += 1
    current = read_jsonl(annotations / "current58.jsonl")
    require(current == [r for r in rows if r["current_experiment_pair"]], "当前实验子集不一致")
    source_key = next(
        p for p in manifest["sources"] if Path(p).as_posix().endswith("051500Z/human_review_pending.jsonl")
    )
    pending = read_jsonl(root / source_key)
    require(
        {(r["query_id"], r["job_key"]) for r in current} == {(r["query_id"], r["job_key"]) for r in pending},
        "当前候选未全标",
    )
    expected_by_pair = {(r["query_id"], r["job_key"]): r for r in pending}
    for row in current:
        original = expected_by_pair[row["query_id"], row["job_key"]]
        require(all(row[k] == original[k] for k in ("query", "title", "jd_text")), "冻结查询/JD被改写")

    human = read_jsonl(root / "data/eval/review_human_e09ee0b905fb/annotations.jsonl")
    labels = {(r["query_id"], r["job_key"]): r["label"] for r in human if r["kind"] == "relevance"}
    require(not set(labels) & {(r["query_id"], r["job_key"]) for r in current}, "新标签覆盖旧人工标签")
    labels.update({(r["query_id"], r["job_key"]): r["label"] for r in current})
    experiment = root / source_key
    rankings = [r for r in read_jsonl(experiment.parent / "per_query.jsonl") if r["kind"] == "human"]
    saved = {(r["query_id"], r["variant"]): r for r in read_jsonl(annotations / "per_query_metrics.jsonl")}
    require(len(rankings) == len(saved), "补充计分条数不一致")
    for row in rankings:
        qid = row["query_id"]
        grades = [labels[qid, key] for key in row["ranking"][:10]]
        pool = [grade for (q, key), grade in labels.items() if q == qid]

        def dcg(items):
            return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(items))

        positives = sum(g == 2 for g in pool)
        ideal = dcg(sorted(pool, reverse=True)[:10])
        measured = dict(
            p10=sum(g == 2 for g in grades) / 10,
            pool_recall10=sum(g == 2 for g in grades) / positives if positives else None,
            pool_ndcg10=dcg(grades) / ideal if ideal else None,
        )
        target = saved[qid, row["variant"]]
        for name, value in measured.items():
            require(
                (value is None and target[name] is None)
                or (
                    value is not None
                    and target[name] is not None
                    and math.isclose(value, target[name], abs_tol=1e-12)
                ),
                f"指标不一致：{qid}/{row['variant']}/{name}",
            )

    frozen = read_json(queries / "manifest.json")
    for name, expected in frozen["hashes"].items():
        require(checksum(queries / name) == expected, f"冻结查询文件变化：{name}")
    records = read_jsonl(queries / "queries.jsonl")
    require(len(records) == frozen["queries"], "查询数量不一致")

    def normalize(text):
        return "".join(text.split()).casefold()

    require(len({r["query_id"] for r in records}) == len(records), "查询ID重复")
    require(len({normalize(r["query"]) for r in records}) == len(records), "查询文本重复")
    old = read_json(root / "data/eval/expanded_probes_v1/queries.json")
    require(
        not {normalize(r["query"]) for r in old} & {normalize(r["query"]) for r in records},
        "与旧查询完全重复",
    )
    groups = defaultdict(set)
    for row in records:
        groups[row["group"]].add(row["split"])
        require(
            row["anchor_job"] is None and row["relevance_labels_status"] == "not_judged",
            "新查询不能冒充已标注",
        )
    require(all(len(split) == 1 for split in groups.values()), "同一意图组跨开发/测试")
    require(dict(Counter(r["split"] for r in records)) == frozen["splits"], "split数量不一致")
    for split in ("dev", "test"):
        require(
            read_jsonl(queries / f"{split}.jsonl") == [r for r in records if r["split"] == split],
            "split子文件不一致",
        )
    return dict(
        status="passed",
        assistant_pairs=len(rows),
        final_experiment_pairs=len(current),
        jd_evidence_lines=evidence_count,
        rescored_rankings=len(rankings),
        queries=len(records),
        splits=frozen["splits"],
        groups=len(groups),
        source="assistant; mixed-label scoring, not new human gold",
        external_model_calls=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = verify(
        args.root,
        args.root / "data/eval/review_assistant_supplement_v1",
        args.root / "data/eval/scenario_intents_v2",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
