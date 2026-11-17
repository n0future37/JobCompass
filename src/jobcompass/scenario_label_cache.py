"""从已缓存模型输出恢复合法方向标签，不重新调用模型或更改原标签。"""

import argparse
import json
from pathlib import Path

from .preprocessing import digest
from .scenario_judging import make_batches, read_jsonl


def recover(directory):
    checkpoint = json.loads((directory / "judge/checkpoint.json").read_bytes())
    if checkpoint["status"] == "running":
        raise ValueError("判分仍在运行，不能读取半成品")
    pairs = {p["pair_id"]: p for b in make_batches(directory) for p in b}
    original = read_jsonl(directory / "judge/annotations.jsonl")
    adopted = {
        r["pair_id"]: dict(
            r,
            strict_usable=r.get("strict_usable", True) and r.get("grade") is not None,
            recovery="original_validated",
        )
        for r in original
    }
    recovered = 0
    for call in checkpoint["calls"]:
        if not call.get("raw_output"):
            continue
        try:
            data = json.loads(call["raw_output"])
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            continue
        counts = {}
        for item in data["items"]:
            if isinstance(item, dict) and isinstance(item.get("pair_id"), str):
                counts[item["pair_id"]] = counts.get(item["pair_id"], 0) + 1
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            identifier = item.get("pair_id")
            if (
                not isinstance(identifier, str)
                or identifier not in pairs
                or identifier in adopted
                or counts.get(identifier) != 1
            ):
                continue
            pair = pairs[identifier]
            job = pair["job"]
            lines = job["jd_text"].split("\n")
            numbers = item.get("lines")
            grade = item.get("grade")
            direction = item.get("direction_grade")
            status = item.get("hard_status")
            if type(direction) is not int or direction not in {0, 1, 2}:
                continue
            if (
                not isinstance(numbers, list)
                or not 1 <= len(numbers) <= 6
                or any(type(n) is not int or not 1 <= n <= len(lines) for n in numbers)
                or len(set(numbers)) != len(numbers)
            ):
                continue
            reason = item.get("reason")
            if not isinstance(reason, str) or not 4 <= len(reason) <= 300:
                continue
            usable = (
                type(grade) is int
                and grade in {0, 1, 2}
                and grade <= direction
                and status in {"satisfied", "violated", "unknown", "not_applicable"}
                and not (status == "violated" and grade != 0)
                and not (status == "unknown" and grade > 1)
            )
            adopted[identifier] = dict(
                pair_id=identifier,
                query_id=pair["query_id"],
                job_key=job["job_key"],
                content_hash=job["content_hash"],
                direction_grade=direction,
                grade=grade if usable else None,
                raw_grade=grade,
                hard_status=status,
                strict_usable=usable,
                reason=reason,
                lines=numbers,
                evidence=[dict(line=n, text=lines[n - 1]) for n in numbers],
                annotation_kind="model",
                annotator=checkpoint["model"],
                batch_id=call["batch_id"],
                recovery="cached_response_direction.v1",
            )
            recovered += 1
    assistant_file = directory / "judge/assistant_completions.jsonl"
    if assistant_file.exists():
        for row in read_jsonl(assistant_file):
            identifier = row["pair_id"]
            if identifier not in pairs or identifier in adopted or row["annotation_kind"] != "assistant":
                raise ValueError("助手补标身份重复或越界")
            job = pairs[identifier]["job"]
            if row["content_hash"] != job["content_hash"] or row["query_id"] != pairs[identifier]["query_id"]:
                raise ValueError("助手补标岗位版本不符")
            if type(row["direction_grade"]) is not int or row["direction_grade"] not in {0, 1, 2}:
                raise ValueError("助手方向等级不合法")
            for e in row["evidence"]:
                if (
                    not 1 <= e["line"] <= len(job["jd_text"].split("\n"))
                    or job["jd_text"].split("\n")[e["line"] - 1] != e["text"]
                ):
                    raise ValueError("助手补标引用不合法")
            adopted[identifier] = dict(row, recovery="assistant_completion.v1")
    rows = [adopted[k] for k in sorted(adopted)]
    output = directory / "judge/direction_annotations.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    missing = [
        dict(
            pair_id=k,
            query_id=p["query_id"],
            query=p["query"]["query"],
            hard_reference=p["query"]["hard_constraints_reference"],
            job_key=p["job"]["job_key"],
            job=p["job"],
        )
        for k, p in pairs.items()
        if k not in adopted
    ]
    (directory / "judge/direction_unjudged.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in missing), encoding="utf-8"
    )
    meta = dict(
        protocol="cached-direction-recovery.v1",
        original_validated=len(original),
        recovered_direction_only_or_extended_evidence=recovered,
        assistant_completions=sum(r["annotation_kind"] == "assistant" for r in rows),
        direction_labeled=len(rows),
        total_pairs=len(pairs),
        direction_unjudged=len(missing),
        strict_usable=sum(r["strict_usable"] for r in rows),
        external_calls=0,
        model=checkpoint["model"],
        method="Keep original accepted records. For still-missing pairs adopt first cached well-formed direction assessment with1..6 real JD evidence lines. Inconsistent strict grade stays null; no label inferred from keywords or coerced to zero.",
        input_hashes={
            n: digest((directory / n).read_bytes())
            for n in ["per_query.jsonl", "judge/checkpoint.json", "judge/annotations.jsonl"]
        },
        output_sha256=digest(output.read_bytes()),
        limits="Model ignores unspecified qualifications in some cases; direction and strict matching must be separately reported. This recovery is structural, not independent semantic gold verification.",
    )
    if assistant_file.exists():
        meta["input_hashes"]["judge/assistant_completions.jsonl"] = digest(assistant_file.read_bytes())
    (directory / "judge/direction_recovery.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.directory), ensure_ascii=False, indent=2))
