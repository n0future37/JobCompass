from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from jobcompass.tasks import TaskQueue


def failure(code):
    return OperationalError("claim", {}, Exception(code, "fixed failure"))


def test_claim_retries_transaction_deadlock_with_backoff(monkeypatch):
    queue = TaskQueue(SimpleNamespace())
    calls, sleeps = [], []

    def claim_once():
        calls.append(1)
        if len(calls) < 3:
            raise failure(1213 if len(calls) == 1 else 1205)
        return {"id": "task", "attempts": 1}

    monkeypatch.setattr(queue, "_claim_once", claim_once)
    monkeypatch.setattr("jobcompass.tasks.time.sleep", sleeps.append)
    assert queue.claim()["id"] == "task"
    assert len(calls) == 3 and sleeps == [0.02, 0.04]


@pytest.mark.parametrize("code,expected_calls", [(1213, 3), (1045, 1)])
def test_claim_exhausts_retry_budget_and_does_not_hide_other_errors(monkeypatch, code, expected_calls):
    queue = TaskQueue(SimpleNamespace())
    calls = []

    def claim_once():
        calls.append(1)
        raise failure(code)

    monkeypatch.setattr(queue, "_claim_once", claim_once)
    monkeypatch.setattr("jobcompass.tasks.time.sleep", lambda _: None)
    with pytest.raises(OperationalError):
        queue.claim()
    assert len(calls) == expected_calls
