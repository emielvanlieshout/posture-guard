"""The download step, which is the one place this project touches a network.

No test here opens a socket: urlopen is stubbed throughout. What is being
checked is the behaviour around the download -- that an existing model is left
alone, that a truncated one is never installed, and that a TLS failure produces
instructions rather than a traceback.
"""

from __future__ import annotations

import ssl
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from posture_guard import model as model_mod
from posture_guard.model import (
    MIN_BYTES,
    MODEL_URL,
    ModelDownloadError,
    ensure_model,
    manual_download_hint,
    model_digest,
    ssl_context,
)

PAYLOAD = b"x" * (MIN_BYTES + 10)


class FakeResponse(BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stub_urlopen(payload=PAYLOAD, status=200, raises=None):
    calls = []

    def fake(url, timeout=None, context=None):
        calls.append({"url": url, "timeout": timeout, "context": context})
        if raises is not None:
            raise raises
        response = FakeResponse(payload)
        response.status = status
        return response

    fake.calls = calls
    return fake


class TestSslContext:
    def test_it_is_a_verifying_context(self):
        ctx = ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode is ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_it_loads_certificates(self):
        assert ssl_context().cert_store_stats()["x509_ca"] > 0

    def test_it_still_works_without_certifi(self, monkeypatch):
        """A machine whose Python does have a system bundle must not be broken."""
        import builtins

        real_import = builtins.__import__

        def no_certifi(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("no certifi")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_certifi)
        assert isinstance(ssl_context(), ssl.SSLContext)


class TestEnsureModel:
    def test_it_downloads_when_missing(self, tmp_path, monkeypatch):
        fake = stub_urlopen()
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake)

        path = ensure_model(tmp_path / "nested" / "pose.task")
        assert path.exists()
        assert path.stat().st_size == len(PAYLOAD)
        assert fake.calls[0]["url"] == MODEL_URL
        assert isinstance(fake.calls[0]["context"], ssl.SSLContext), "must verify TLS"

    def test_an_existing_model_is_left_alone(self, tmp_path, monkeypatch):
        path = tmp_path / "pose.task"
        path.write_bytes(PAYLOAD)
        fake = stub_urlopen()
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake)

        ensure_model(path)
        assert not fake.calls, "no reason to download it twice"

    def test_force_downloads_again(self, tmp_path, monkeypatch):
        path = tmp_path / "pose.task"
        path.write_bytes(PAYLOAD)
        fake = stub_urlopen()
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake)

        ensure_model(path, force=True)
        assert len(fake.calls) == 1

    def test_a_truncated_file_is_replaced_not_trusted(self, tmp_path, monkeypatch):
        path = tmp_path / "pose.task"
        path.write_bytes(b"half a model")
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", stub_urlopen())

        ensure_model(path)
        assert path.stat().st_size == len(PAYLOAD)

    def test_a_truncated_download_is_never_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            model_mod.urllib.request, "urlopen", stub_urlopen(payload=b"nope")
        )
        path = tmp_path / "pose.task"
        with pytest.raises(ModelDownloadError, match="stopped after"):
            ensure_model(path)
        assert not path.exists(), "a partial file must not be left behind"

    def test_no_partial_file_survives_a_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            model_mod.urllib.request,
            "urlopen",
            stub_urlopen(raises=urllib.error.URLError("boom")),
        )
        path = tmp_path / "pose.task"
        with pytest.raises(ModelDownloadError):
            ensure_model(path)
        assert list(tmp_path.iterdir()) == []


class TestErrorMessages:
    def test_a_certificate_failure_says_what_to_do(self, tmp_path, monkeypatch):
        """The exact failure a fresh macOS python.org install produces."""
        error = urllib.error.URLError(
            ssl.SSLCertVerificationError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                "unable to get local issuer certificate"
            )
        )
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", stub_urlopen(raises=error))

        with pytest.raises(ModelDownloadError) as caught:
            ensure_model(tmp_path / "pose.task")

        message = str(caught.value)
        assert "could not be verified" in message
        assert "pip install --upgrade certifi" in message
        assert "Install Certificates.command" in message
        assert "curl" in message, "and a way out that does not depend on Python's TLS"

    def test_an_offline_machine_gets_a_different_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            model_mod.urllib.request,
            "urlopen",
            stub_urlopen(raises=urllib.error.URLError("Name or service not known")),
        )
        with pytest.raises(ModelDownloadError, match="could not reach the model host"):
            ensure_model(tmp_path / "pose.task")

    def test_a_bad_status_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_mod.urllib.request, "urlopen", stub_urlopen(status=503))
        with pytest.raises(ModelDownloadError, match="status 503"):
            ensure_model(tmp_path / "pose.task")

    def test_the_manual_hint_is_a_runnable_command(self, tmp_path):
        hint = manual_download_hint(tmp_path / "Application Support" / "pose.task")
        assert hint.startswith("curl -L --create-dirs -o ")
        assert MODEL_URL in hint
        # The default path contains spaces on macOS, so both arguments are quoted.
        assert hint.count('"') == 4


def test_digest_is_stable(tmp_path):
    path = tmp_path / "pose.task"
    path.write_bytes(PAYLOAD)
    assert model_digest(path) == model_digest(path)
    assert len(model_digest(path)) == 64
