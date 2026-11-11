import pytest

from raptor.models.keypool import KeyPool, is_rate_limit, is_retryable
from raptor.settings import Settings


class FakeAPIError(Exception):
    def __init__(self, code: int, msg: str = "") -> None:
        super().__init__(f"{code} {msg}")
        self.code = code


def pool(n: int = 3) -> KeyPool:
    return KeyPool([f"k{i}" for i in range(n)], client_factory=lambda key: key)


def test_round_robin_uses_every_key_before_waiting():
    p = pool(3)
    used = [p.call("llm:x", 1, lambda client: client) for _ in range(3)]
    assert sorted(used) == ["k0", "k1", "k2"]  # rpm=1 per key: three calls, three keys, no wait
    assert p.calls_per_key == [1, 1, 1]


def test_rate_limited_key_fails_over_to_next():
    p = pool(2)
    seen = []

    def fn(client):
        seen.append(client)
        if client == "k0":
            raise FakeAPIError(429, "RESOURCE_EXHAUSTED")
        return "ok"

    assert p.call("llm:x", None, fn) == "ok"
    assert seen == ["k0", "k1"]
    # k0 is cooling down for this bucket, so the next calls go to k1 ...
    assert p.call("llm:x", None, lambda c: c) == "k1"
    # ... but a different model bucket can still use k0.
    assert p.call("embed:y", None, lambda c: c) == "k0"


def test_non_retryable_error_is_raised_immediately():
    p = pool(2)
    calls = []

    def fn(client):
        calls.append(client)
        raise FakeAPIError(400, "INVALID_ARGUMENT")

    with pytest.raises(FakeAPIError):
        p.call("llm:x", None, fn)
    assert len(calls) == 1


def test_error_classification():
    assert is_rate_limit(FakeAPIError(429))
    assert is_retryable(FakeAPIError(503))
    assert not is_retryable(FakeAPIError(400))
    with pytest.raises(ValueError):
        KeyPool([], client_factory=lambda k: k)


def test_settings_collects_numbered_keys(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=a\nGEMINI_API_KEY3=c\nGEMINI_API_KEY2=b\nOTHER=x\n")
    monkeypatch.setenv("GEMINI_API_KEY10", "j")
    settings = Settings(_env_file=None)
    assert settings.gemini_keys(env_file=env) == ["a", "b", "c", "j"]
    monkeypatch.setenv("GEMINI_API_KEY2", "a")  # duplicates are dropped
    assert settings.gemini_keys(env_file=env) == ["a", "c", "j"]
    assert Settings(_env_file=None).gemini_keys(env_file=None) == ["a", "j"]
