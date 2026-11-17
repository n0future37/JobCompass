"""固定职责证据参考评测：只在dev选门槛，test使用已锁配置；不改原始标签。"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from .conditions import SearchRequest
from .evidence import EvidenceGuard
from .preprocessing import digest
from .settings import Settings


def load(dataset, split):
    if split == "acceptance":
        meta = json.loads((dataset / "acceptance_manifest.json").read_bytes())
        body = (dataset / "acceptance.jsonl").read_bytes()
        if digest(body) != meta["sha256"]:
            raise ValueError("补充验收指纹不符")
        raw = [json.loads(x) for x in body.decode("utf-8").split("\n") if x]
        jobs = {r["job"]["job_key"]: r["job"] for r in raw}
        rows = [{k: v for k, v in r.items() if k != "job"} | {"job_key": r["job"]["job_key"]} for r in raw]
        return rows, jobs
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    for name in ("jobs.jsonl", split + ".jsonl"):
        if digest((dataset / name).read_bytes()) != manifest["hashes"][name]:
            raise ValueError("证据参考文件指纹不符")

    def readl(name):
        return [json.loads(s) for s in (dataset / name).read_text(encoding="utf-8").split("\n") if s]

    jobs = {j["job_key"]: j for j in readl("jobs.jsonl")}
    records = readl(split + ".jsonl")
    if len(records) != manifest["splits"][split]:
        raise ValueError("参考数量不符")
    return records, jobs


def confusion(rows, threshold, direction_threshold):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = (
            row["max_score"] is not None
            and row["max_score"] >= threshold
            and row["direction_score"] >= direction_threshold
        )
        if predicted and row["expected_supported"]:
            tp += 1
        elif predicted:
            fp += 1
        elif row["expected_supported"]:
            fn += 1
        else:
            tn += 1
    return dict(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        positive_retention=tp / (tp + fn) if tp + fn else None,
        false_approval_rate=fp / (fp + tn) if fp + tn else None,
    )


def run(dataset, split, output, calibrate=False):
    if calibrate and split != "dev":
        raise ValueError("只允许开发集选择门槛")
    records, jobs = load(dataset, split)
    settings = Settings()
    guard = EvidenceGuard(settings)
    rows = []
    started = perf_counter()
    for row in records:
        job = jobs[row["job_key"]]
        result = guard.prepare(
            [dict(job=job, hard_condition_checks=[])], SearchRequest(query=row["query"], generate=False)
        )
        check = result["checks"][0]
        if result["errors"]:
            raise RuntimeError("本地职责模型不可用")
        rows.append(
            row
            | dict(
                max_score=check["max_score"],
                direction_score=check["direction_score"],
                check=check,
                focus=result["focus"],
            )
        )
    threshold = guard.spec["threshold"]
    direction_threshold = guard.spec["direction_threshold"]
    selected = None
    if calibrate:

        def grid(field):
            values = [r[field] for r in rows if r[field] is not None]
            return sorted(
                {-12.0, 12.0}
                | {
                    point
                    for v in values
                    for point in (math.floor(v * 10) / 10, math.floor(v * 10) / 10 + 0.1)
                }
            )

        choices = []
        for line_threshold in grid("max_score"):
            for context_threshold in grid("direction_score"):
                counts = confusion(rows, line_threshold, context_threshold)
                choices.append(
                    (counts["fp"] == 0, counts["tp"], -counts["fp"], line_threshold, context_threshold)
                )
        choice = max(choices)
        threshold, direction_threshold = choice[-2:]
        selected = dict(
            threshold=threshold,
            direction_threshold=direction_threshold,
            criterion="dev only: zero false approvals, maximize positive retention, then stricter line/context thresholds; raw logits not probabilities",
            dataset_manifest_sha256=digest((dataset / "manifest.json").read_bytes()),
        )
    summary = dict(
        split=split,
        references=len(rows),
        threshold=threshold,
        direction_threshold=direction_threshold,
        counts=confusion(rows, threshold, direction_threshold),
        calibration=selected,
        reference_kind="assistant references, public JD and fictional controls; not human blind gold",
        seconds=perf_counter() - started,
        external_calls=0,
        source_hashes={
            name: digest(Path(__file__).with_name(name).read_bytes()) for name in ("evidence.py", "models.py")
        },
        manifest_sha256=digest((dataset / "manifest.json").read_bytes()),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / (split + "_scored.jsonl")).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (output / (split + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test", "acceptance"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.dataset, args.split, args.output, args.calibrate), ensure_ascii=False), flush=True
    )
