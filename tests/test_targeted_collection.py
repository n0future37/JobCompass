import json

from jobcompass.crawlers.zhilian import resolve_city, select_card


def test_selection_keeps_off_topic_titles_but_requires_city_type_and_new_identity():
    target = dict(city="北京", employment_types=["实习"], title_any=["Python", "后端"])
    card = dict(number="CC1J2", workCity="北京", workType="实习", name="Python后端实习")
    assert select_card(card, target, set())[0]
    assert not select_card(card, target, {"zhilian:CC1J2"})[0]
    assert not select_card(card | {"workCity": "深圳"}, target, set())[0]
    assert not select_card(card | {"workType": "校园"}, target, set())[0]
    for title in ["纯销售", "AI产品经理", "射频硬件开发实习生"]:
        assert select_card(card | {"name": title}, target, set())[0]
    assert select_card(card, {"city": "北京", "employment_types": ["实习"]}, set())[0]
    assert not select_card(card | {"number": "../bad"}, target, set())[0]
    assert resolve_city({"hotCity": [{"name": "北京", "code": "530"}]}, "北京") == "530"


def test_verification_stops_entire_collection_and_keeps_raw(tmp_path, monkeypatch):
    from jobcompass.crawlers import zhilian

    root = tmp_path
    snapshot = root / "data/processed/old"
    snapshot.mkdir(parents=True)
    (snapshot / "jobs.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(zhilian, "ROOT", root)
    monkeypatch.setattr(zhilian, "RAW", root / "data/raw")
    monkeypatch.setattr(zhilian, "RUN_ID", "unit")
    responses = [
        {"code": 200, "data": {"hotCity": [{"name": "北京", "code": "530"}]}},
        {
            "code": 200,
            "data": {
                "list": [{"number": "CC1J2", "workCity": "北京", "workType": "实习", "name": "Python后端"}]
            },
        },
        {"code": 200, "data": {"isVerification": True}},
    ]
    calls = []

    class FakeProbe:
        def __init__(self, source):
            self.directory = root / "data/raw/2026-11-03/zhilian/unit"
            self.directory.mkdir(parents=True)

        def request(self, label, url, payload=None, headers=None):
            calls.append(label)
            body = json.dumps(responses[len(calls) - 1]).encode()
            path = self.directory / (label + ".body")
            path.write_bytes(body)
            meta = {"status": 200, "collected_at": "2026-11-03T01:00:00+00:00"}
            path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
            return body, meta, path.relative_to(root).as_posix()

    monkeypatch.setattr(zhilian, "Probe", FakeProbe)
    target = dict(
        id="one", city="北京", keyword="Python 实习", employment_types=["实习"], title_any=["Python"], quota=1
    )
    plan = dict(
        targets=[target, target | {"id": "two"}],
        max_new_jobs=2,
        max_pages_per_target=1,
        max_detail_attempts=2,
        existing_snapshot="data/processed/old",
    )
    path = root / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    result = zhilian.collect_selected(path)
    assert result["status"] == "stopped" and result["accepted"] == []
    assert len(calls) == 3 and result["detail_attempts"] == 1
    assert len(list((root / "data/raw").rglob("*.body"))) == 3
    assert (root / "artifacts/collections/unit/checkpoint.json").is_file()


def test_selected_snapshot_excludes_unapproved_raw_but_keeps_original(tmp_path):
    from jobcompass.preprocessing import preprocess

    raw = tmp_path / "data/raw/2026-11-03/zhilian/b"
    raw.mkdir(parents=True)
    for name in ["accepted", "not_selected"]:
        p = raw / (name + ".body")
        p.write_text(
            json.dumps(
                {
                    "code": 200,
                    "data": {
                        "detailedPosition": {
                            "number": name,
                            "positionName": "Python开发",
                            "jobDesc": "开发接口",
                            "workCity": "北京",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        p.with_suffix(".meta.json").write_text(
            json.dumps({"status": 200, "collected_at": "2026-11-03T01:00:00+00:00"}), encoding="utf-8"
        )
    selected = ["data/raw/2026-11-03/zhilian/b/accepted.body"]
    manifest = preprocess(tmp_path / "data/raw", tmp_path / "out", allowed_raw_files=selected)
    assert manifest["jobs"] == 1 and manifest["skipped"] == 1
    assert manifest["raw_selection"]["selected_raw_files"] == selected
    assert (raw / "not_selected.body").is_file()
    import pytest

    with pytest.raises(ValueError, match="不存在或越界"):
        preprocess(tmp_path / "data/raw", tmp_path / "bad", allowed_raw_files=["data/raw/../../outside.body"])
    assert not (tmp_path / "bad").exists()


def test_prior_batches_require_complete_source_and_unchanged_raw(tmp_path, monkeypatch):
    import pytest

    from jobcompass.crawlers import zhilian
    from jobcompass.preprocessing import digest

    monkeypatch.setattr(zhilian, "ROOT", tmp_path)
    monkeypatch.setattr(zhilian, "RAW", tmp_path / "data/raw")
    raw = tmp_path / "data/raw/2026-11-03/zhilian/one/detail.body"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"source-body")
    prior = tmp_path / "artifacts/collections/one/checkpoint.json"
    prior.parent.mkdir(parents=True)
    record = dict(
        source="zhilian",
        status="complete",
        accepted=[
            dict(
                job_key="zhilian:CC1J2",
                raw_file=raw.relative_to(tmp_path).as_posix(),
                body_sha256=digest(raw.read_bytes()),
            )
        ],
    )
    prior.write_text(json.dumps(record), encoding="utf-8")
    relative = prior.relative_to(tmp_path).as_posix()
    known = zhilian.prior_collection_jobs([relative])
    assert known == {"zhilian:CC1J2"}
    assert select_card(dict(number="CC1J2", workCity="北京"), dict(city="北京"), known)[1] == "known_identity"
    record["status"] = "stopped"
    prior.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="未完成"):
        zhilian.prior_collection_jobs([relative])
    record["status"] = "complete"
    prior.write_text(json.dumps(record), encoding="utf-8")
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="指纹变化"):
        zhilian.prior_collection_jobs([relative])
    with pytest.raises(ValueError, match="越界"):
        zhilian.prior_collection_jobs(["outside.json"])
