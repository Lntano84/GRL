"""Four bounded Range streams, reusing the serial prefix without redownload."""
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests
from acquire import ARCHIVE, DATA, EXPECTED, MAX_BODY, MAX_WALL, OUT, STATE, URL

PSTATE = OUT / "download_parallel_state.json"
PARTS = DATA / "range_parts"
LOCK = threading.Lock()
BLOCK = 16 * 1024**2


def main():
    serial = json.loads(STATE.read_text())
    if PSTATE.exists():
        state = json.loads(PSTATE.read_text())
    else:
        state = {"source": URL, "etag": serial["etag"], "serial_prefix_bytes": ARCHIVE.stat().st_size,
                 "body_bytes_received": serial["body_bytes_received"], "workers": 4,
                 "first_started_epoch": ARCHIVE.stat().st_birthtime, "parts": {}, "complete": False,
                 "reason": "Serial transfer too slow; retain exact prefix and use four disjoint ranges."}
    if state["complete"]:
        print("Already complete", flush=True)
        return
    PARTS.mkdir(parents=True, exist_ok=True)
    deadline = state["first_started_epoch"] + MAX_WALL

    def save():
        temp = PSTATE.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # Windows readers may briefly deny replacement while holding the old JSON.
        # Retry the rename only; never repeat a received network block.
        for attempt in range(30):
            try:
                os.replace(temp, PSTATE)
                break
            except PermissionError:
                if attempt==29:
                    raise
                time.sleep(0.1)

    with requests.get(URL, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}, stream=True, timeout=(20,40)) as r:
        assert r.status_code == 206 and r.headers.get("Content-Range") == f"bytes 0-0/{EXPECTED}"
        assert r.headers.get("ETag") == state["etag"]
        signed_url = r.url
        assert r.raw.read(1) == b"B"
        state["body_bytes_received"] += 1
    save()

    def fetch(bounds):
        start, end = bounds
        path = PARTS / f"{start:012d}-{end:012d}.part"
        have = path.stat().st_size if path.exists() else 0
        length = end-start+1
        assert 0 <= have <= length
        if have == length:
            return
        if time.time() > deadline:
            raise RuntimeError("Global 2-hour download ceiling")
        with LOCK:
            if state["body_bytes_received"] + length-have > MAX_BODY:
                raise RuntimeError("Cumulative response-body ceiling")
        headers = {"Range": f"bytes={start+have}-{end}", "If-Match": state["etag"], "Accept-Encoding": "identity"}
        with requests.get(signed_url, headers=headers, stream=True, timeout=(20,90)) as r:
            assert r.status_code == 206 and r.headers.get("Content-Range") == f"bytes {start+have}-{end}/{EXPECTED}"
            assert r.headers.get("ETag") == state["etag"]
            with path.open("ab") as f:
                for chunk in r.iter_content(1024**2):
                    if not chunk:
                        continue
                    if time.time() > deadline:
                        raise RuntimeError("Global 2-hour download ceiling")
                    assert f.tell()+len(chunk) <= length
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    with LOCK:
                        state["body_bytes_received"] += len(chunk)
                        if state["body_bytes_received"] > MAX_BODY:
                            raise RuntimeError("Cumulative response-body ceiling")
                        state["parts"][path.name] = f.tell()
                        save()
        assert path.stat().st_size == length

    ranges = [(x, min(x+BLOCK-1, EXPECTED-1)) for x in range(state["serial_prefix_bytes"], EXPECTED, BLOCK)]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, x) for x in ranges]
        for future in cf.as_completed(futures):
            future.result()
            completed = state["serial_prefix_bytes"] + sum(state["parts"].values())
            print(json.dumps({"saved_bytes": completed, "total": EXPECTED, "percent": round(100*completed/EXPECTED,2), "elapsed_s": round(time.time()-state["first_started_epoch"],1)}), flush=True)
    # Never overwrite the partial prefix until a complete independently hashed assembly exists.
    assembly = DATA / "l2rpn_idf_2023.assembled.tar.bz2"
    sha = hashlib.sha256()
    with assembly.open("wb") as dst:
        for source in [ARCHIVE] + [PARTS / f"{a:012d}-{b:012d}.part" for a,b in ranges]:
            with source.open("rb") as src:
                for chunk in iter(lambda: src.read(8*1024**2), b""):
                    dst.write(chunk)
                    sha.update(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    assert assembly.stat().st_size == EXPECTED
    os.replace(assembly, ARCHIVE)
    state.update(complete=True, saved_bytes=EXPECTED, sha256=sha.hexdigest(), completed_epoch=time.time())
    save()
    print(json.dumps({"complete": True, "sha256": sha.hexdigest(), "body_bytes_received":state["body_bytes_received"]}),flush=True)


if __name__ == "__main__":
    main()
