"""Store the secret code as a salted hash.
  python -m src.code_store 2 5 3        # set the code (3 distinct digits 1-5)"""
import hashlib
import json
import os
import sys

from . import config


def _hash(salt: bytes, digits) -> str:
    return hashlib.pbkdf2_hmac("sha256", bytes(digits), salt, 100_000).hex()


def set_code(digits):
    assert len(digits) == config.CODE_LEN and len(set(digits)) == len(digits)
    assert all(1 <= d <= 5 for d in digits)
    salt = os.urandom(16)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.CODE_PATH.write_text(json.dumps({"salt": salt.hex(), "hash": _hash(salt, digits)}))


def verify(digits) -> bool:
    d = json.loads(config.CODE_PATH.read_text())
    return _hash(bytes.fromhex(d["salt"]), digits) == d["hash"]


if __name__ == "__main__":
    set_code([int(a) for a in sys.argv[1:]])
    print("code saved")
