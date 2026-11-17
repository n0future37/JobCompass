"""本地条件解析评测：只读指定split，参考为预先撰写的助手条件，不冒充人类金标。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .conditions import SearchRequest
from .intent import HardConditions, parse_intent
from .preprocessing import digest


def evaluate(dataset, split, output):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    name = split + ".jsonl"
    if digest((dataset / name).read_bytes()) != manifest["hashes"][name]:
        raise ValueError("冻结条件查询指纹不符")
    records = [json.loads(x) for x in (dataset / name).read_text(encoding="utf-8").split("\n") if x]
    if len(records) != manifest["splits"][split]:
        raise ValueError("split数量不符")
    rows, tp, fp, fn = [], 0, 0, 0
    for row in records:
        result = parse_intent(SearchRequest(query=row["query"], **row["request"]))
        actual = result.conditions()
        expected = HardConditions.model_validate(row["expected_conditions"]).model_dump()
        a = {(f, v) for f, values in actual.items() for v in values}
        e = {(f, v) for f, values in expected.items() for v in values}
        tp += len(a & e)
        fp += len(a - e)
        fn += len(e - a)
        for span in result.sources:
            if span.source == "query" and row["query"][span.start : span.end] != span.quote:
                raise ValueError("来源span不是查询原文")
        soft_ok = all(
            set(result.soft_preferences.get(f, [])) == set(values)
            for f, values in row["expected_soft"].items()
        )
        hard_ok = a == e
        status_ok = result.status == row["expected_status"]
        rows.append(
            dict(
                query_id=row["query_id"],
                category=row["category"],
                query=row["query"],
                request=row["request"],
                expected=expected,
                expected_status=row["expected_status"],
                intent=result.public(),
                hard_exact=hard_ok,
                status_exact=status_ok,
                soft_exact=soft_ok,
                passed=hard_ok and status_ok and soft_ok,
            )
        )
    supported = [r for r in rows if r["expected_status"] == "ready"]
    summary = dict(
        protocol="intent-reference-eval.v1",
        split=split,
        queries=len(rows),
        reference_kind="assistant-authored pre-implementation references, not human/independent blind gold",
        passed=sum(r["passed"] for r in rows),
        hard_exact=sum(r["hard_exact"] for r in rows),
        status_exact=sum(r["status_exact"] for r in rows),
        soft_exact=sum(r["soft_exact"] for r in rows),
        supported_queries=len(supported),
        false_rejections=sum(r["intent"]["status"] != "ready" for r in supported),
        missed_rejections=sum(
            r["expected_status"] != "ready" and r["intent"]["status"] == "ready" for r in rows
        ),
        slot_precision=tp / (tp + fp) if tp + fp else None,
        slot_recall=tp / (tp + fn) if tp + fn else None,
        slot_counts=dict(tp=tp, fp=fp, fn=fn),
        actual_statuses=dict(Counter(r["intent"]["status"] for r in rows)),
        manifest_sha256=digest((dataset / "manifest.json").read_bytes()),
        parser_sha256=digest(Path(__file__).with_name("intent.py").read_bytes()),
        external_calls=0,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / f"{split}_details.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.split, args.output), ensure_ascii=False))
