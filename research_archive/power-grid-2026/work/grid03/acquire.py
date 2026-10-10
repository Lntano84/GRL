"""Bounded, resumable public archive acquisition. TLS verification stays enabled."""
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid03"
DATA = ROOT / "work/grid03/data"
URL = "https://codalab.lisn.upsaclay.fr/my/datasets/download/073fb805-76e5-423d-97e3-83a62a7ac0d0"
EXPECTED = 5358993257
MAX_BODY = 6 * 1024**3
MAX_WALL = 7200
CHUNK = 64 * 1024**2
ARCHIVE = DATA / "l2rpn_idf_2023.tar.bz2"
STATE = OUT / "download_state.json"


def save(obj):
    tmp = STATE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "source": URL, "expected_size": EXPECTED, "tls_verification": True,
        "max_application_body_bytes": MAX_BODY, "body_bytes_received": 0,
        "attempts": [], "complete": False,
    }
    start = time.monotonic()
    free = shutil.disk_usage(DATA).free
    if free < (EXPECTED - (ARCHIVE.stat().st_size if ARCHIVE.exists() else 0)) + 12 * 1024**3:
        raise RuntimeError("Insufficient free space for archive plus extraction reserve")
    session = requests.Session()
    headers = {"Accept-Encoding": "identity"}
    with session.get(URL, headers={**headers, "Range": "bytes=0-0"}, stream=True, timeout=(20, 40)) as r:
        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes 0-0/{EXPECTED}":
            raise RuntimeError(f"Unexpected probe: {r.status_code} {r.headers.get('Content-Range')}")
        etag = r.headers.get("ETag")
        if state.get("etag") is not None and state["etag"] != etag:
            raise RuntimeError("Remote identity changed; do not append")
        signed_url = r.url
        p = urlsplit(signed_url)
        state.update(etag=etag, final_url_without_query=f"{p.scheme}://{p.netloc}{p.path}")
        probe = r.raw.read(1)
        state["body_bytes_received"] += len(probe)
        if probe != b"B":
            raise RuntimeError("Not a bzip archive")
    save(state)
    while (ARCHIVE.stat().st_size if ARCHIVE.exists() else 0) < EXPECTED:
        if time.monotonic() - start > MAX_WALL:
            raise RuntimeError("Acquisition wall ceiling reached; partial archive is resumable")
        pos = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        end = min(EXPECTED - 1, pos + CHUNK - 1)
        if state["body_bytes_received"] + (end - pos + 1) > MAX_BODY:
            raise RuntimeError("Cumulative network payload ceiling reached")
        attempt = {"start": pos, "end": end, "received": 0, "status": "running"}
        state["attempts"].append(attempt)
        save(state)
        try:
            with session.get(signed_url, headers={**headers, "Range": f"bytes={pos}-{end}", "If-Match": etag}, stream=True, timeout=(20, 90)) as r:
                expected_range = f"bytes {pos}-{end}/{EXPECTED}"
                if r.status_code != 206 or r.headers.get("Content-Range") != expected_range or r.headers.get("ETag") != etag:
                    raise RuntimeError(f"Range/identity mismatch: {r.status_code} {r.headers.get('Content-Range')}")
                with ARCHIVE.open("ab") as f:
                    for block in r.iter_content(1024**2):
                        if not block:
                            continue
                        if attempt["received"] + len(block) > end - pos + 1:
                            raise RuntimeError("Response exceeded requested range")
                        f.write(block)
                        f.flush()
                        os.fsync(f.fileno())
                        attempt["received"] += len(block)
                        state["body_bytes_received"] += len(block)
                        state["saved_bytes"] = f.tell()
                        save(state)
                if attempt["received"] != end - pos + 1:
                    raise RuntimeError("Short response")
                attempt["status"] = "complete"
        except requests.RequestException as exc:
            # Save the exact partial prefix. Do not automatically restart a complete download.
            attempt["status"] = "interrupted"
            attempt["error_type"] = type(exc).__name__
            save(state)
            raise
        save(state)
        size = ARCHIVE.stat().st_size
        elapsed = time.monotonic() - start
        print(json.dumps({"bytes": size, "total": EXPECTED, "percent": round(100*size/EXPECTED, 2), "wall_s": round(elapsed, 2)}), flush=True)
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    state.update(complete=True, saved_bytes=ARCHIVE.stat().st_size, sha256=digest.hexdigest(), invocation_wall_s=time.monotonic()-start)
    save(state)
    print(json.dumps({"complete": True, "size": state["saved_bytes"], "sha256": state["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
