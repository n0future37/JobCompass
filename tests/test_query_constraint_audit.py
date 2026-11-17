"""语义评测控制项不泄漏期望，格式失败不算保持。"""

import json

import pytest

from jobcompass import query_constraint_audit as audit


def sample():
    return dict(
        case_id="S_DX",
        split="dev",
        query="只看深圳",
        hard=["深圳"],
        soft=[],
        variants=[dict(variant_id="identity", text="只看深圳"), dict(variant_id="rewrite", text="深圳招聘")],
    )


def test_expected_controls_never_sent():
    case = sample() | dict(expected={"rewrite": "preserved"})
    public = audit.public_case(case)
    assert "expected" not in public and public["variants"] == case["variants"]


def test_semantic_quote_must_belong_to_correct_text():
    case = sample() | dict(variants=[dict(variant_id="rewrite", text="深圳招聘")])
    item = dict(
        variant_id="rewrite",
        status="preserved",
        reason="保留深圳",
        original_quote="深圳",
        transformed_quote="深圳",
    )
    assert audit.validate(dict(items=[item]), case) == [item]
    with pytest.raises(ValueError):
        audit.validate(dict(items=[item | dict(transformed_quote="上海")]), case)


def test_partial_report_preserves_format_unknown(tmp_path, monkeypatch):
    case = sample()
    monkeypatch.setattr(audit, "cases", lambda directory: [case])
    items = [
        dict(
            variant_id="identity",
            status="preserved",
            reason="文本未变",
            original_quote="深圳",
            transformed_quote="深圳",
        ),
        dict(
            variant_id="rewrite",
            status="preserved",
            reason="错误引用",
            original_quote="深圳",
            transformed_quote="上海",
        ),
    ]
    (tmp_path / "semantic_condition_calls.jsonl").write_text(
        json.dumps(dict(case_id="S_DX", raw_output=json.dumps(dict(items=items)), cost=dict(cny=0))) + "\n",
        encoding="utf-8",
    )
    result = audit.report(tmp_path)
    assert result["by_split"]["dev"]["identity"] == {"preserved": 1}
    assert result["by_split"]["dev"]["rewrite"] == {"format_unknown": 1}
