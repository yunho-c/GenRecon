#!/usr/bin/env python3
"""Download the official GenRecon weights with resumable, verified HTTP ranges.

The server supports ranges; bounded concurrent requests avoid a slow single
connection. Completed chunks are retained across interruptions. No Hub token is
sent to this public university host. SHA256 values are recorded, not asserted to
be upstream-published hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

BASE = "https://kaldir.vc.cit.tum.de/genrecon/"


def download(name, dest, workers):
    url = BASE + name + ".pt"
    target = dest / (name + ".pt")
    parts = dest / (name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size, etag = int(r.headers["Content-Length"]), r.headers["ETag"]
    identity = {"url": url, "bytes": size, "etag": etag}
    manifest = parts / "source.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError(f"Source changed; inspect existing partial download: {parts}")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    chunk = 8 * 1024 * 1024
    ranges = [(i, min(i+chunk, size)) for i in range(0, size, chunk)]

    def fetch(bounds):
        start, end = bounds
        part = parts / str(start)
        if part.exists() and part.stat().st_size == end-start:
            return
        for retry in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end-1}", "If-Match": etag})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end-1}/{size}":
                        raise RuntimeError("Server did not honor the requested byte range")
                    data = r.read()
                if len(data) != end-start:
                    raise RuntimeError("Truncated range")
                temp = part.with_suffix(".partial")
                temp.write_bytes(data)
                temp.replace(part)
                return
            except Exception:
                if retry == 3:
                    raise
                time.sleep(2**retry)

    if not target.exists():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, bounds) for bounds in ranges]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                if i % 20 == 0 or i == len(ranges):
                    print(f"{name}: {i}/{len(ranges)} ranges", flush=True)
        with target.with_suffix(".partial").open("wb") as out:
            for start, _ in ranges:
                out.write((parts / str(start)).read_bytes())
        target.with_suffix(".partial").replace(target)
    if target.stat().st_size != size:
        raise RuntimeError(f"Incorrect final size: {target}")
    sha = hashlib.sha256()
    with target.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(data)
    identity["sha256"] = sha.hexdigest()
    target.with_suffix(".source.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=["sparse_structure", "shape_slat", "texture_slat"], default=["sparse_structure", "shape_slat", "texture_slat"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        p.error("Use between 1 and 16 download workers")
    args.destination.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        download(model, args.destination, args.workers)
