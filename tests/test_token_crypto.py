import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings
from app.services import token_crypto


@pytest.fixture(autouse=True)
def _fresh_caches():
    get_settings.cache_clear()
    token_crypto._get_fernet.cache_clear()
    yield
    get_settings.cache_clear()
    token_crypto._get_fernet.cache_clear()


def test_round_trip(monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())

    ciphertext = token_crypto.encrypt_token("access-token-123")

    assert ciphertext != "access-token-123"
    assert token_crypto.decrypt_token(ciphertext) == "access-token-123"


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        token_crypto.encrypt_token("x")


def test_token_encrypted_under_another_key_is_rejected(monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    ciphertext = token_crypto.encrypt_token("secret")

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    token_crypto._get_fernet.cache_clear()

    with pytest.raises(InvalidToken):
        token_crypto.decrypt_token(ciphertext)
