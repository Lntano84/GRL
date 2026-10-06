"""List the shared Dropbox folder behind the LimeQO dataset link, using the public API.

No credentials needed: a shared link can be listed through ``/2/sharing/list_folder``.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

SHARED = "https://www.dropbox.com/scl/fo/y4e88tmcx7ywo4ou1unnh/ABN6iqV1t_ecktO51gsPKRc?rlkey=hedjnmkpak3r3gxjzx48s9etu&st=uxnr4s17&dl=0"
HDR = {"User-Agent": "probe", "Content-Type": "application/json"}


def api(path, payload):
    req = urllib.request.Request("https://api.dropboxapi.com/2/" + path,
                                 data=json.dumps(payload).encode(),
                                 headers={**HDR, "Authorization": "Bearer "}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    try:
        j = api("sharing/list_folder", {"path": SHARED, "shared_link": SHARED})
    except Exception as exc:
        body = getattr(exc, "read", lambda: b"")()
        print(f"  list_folder failed: {exc}")
        if body:
            print(f"  body: {body[:400]!r}")
        return 1
    print(f"  folder: {j.get('name')}   entries: {len(j.get('entries', []))}")
    for e in j.get("entries", []):
        tag = e.get(".tag")
        if tag == "file":
            print(f"    [file] {e['name']:<40} {e.get('size', 0):>14,} bytes")
        else:
            print(f"    [dir ] {e['name']}")
    print(json.dumps(j, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
