#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REGISTRY = "harbor.sourcefind.cn:5443"
BASE_URL = f"https://{REGISTRY}"
REPO = "dcu/admin/base/vllm"
TAG = "0.18.1-ubuntu22.04-dtk26.04-py3.10"
IMAGE_REF = f"{REGISTRY}/{REPO}:{TAG}"
OUT_ROOT = Path("/data/shared/images")
NAME = "vllm_0.18.1-ubuntu22.04-dtk26.04-py3.10"
LAYOUT_DIR = OUT_ROOT / NAME
ARCHIVE = OUT_ROOT / f"{NAME}.oci.tar"

ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.manifest.v1+json"
)


def urlopen(req, timeout=60):
    ctx = ssl._create_unverified_context()
    return urllib.request.urlopen(req, context=ctx, timeout=timeout)


def get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urlopen(req) as resp:
        return json.loads(resp.read())


def get_token():
    scope = urllib.parse.quote(f"repository:{REPO}:pull", safe=":")
    data = get_json(f"{BASE_URL}/service/token?service=harbor-registry&scope={scope}")
    return data["token"]


def fetch_manifest(token):
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    req = urllib.request.Request(f"{BASE_URL}/v2/{REPO}/manifests/{TAG}", headers=headers)
    with urlopen(req) as resp:
        body = resp.read()
        digest = resp.headers.get("Docker-Content-Digest")
        media_type = resp.headers.get("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
    return media_type, digest, body, json.loads(body)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def blob_path(digest):
    algo, value = digest.split(":", 1)
    if algo != "sha256":
        raise ValueError(f"unsupported digest algorithm: {digest}")
    return LAYOUT_DIR / "blobs" / algo / value


def download_blob(token, digest, expected_size):
    dest = blob_path(digest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected_size:
        actual = sha256_file(dest)
        if actual == digest:
            print(f"OK {digest} {expected_size} bytes", flush=True)
            return
        print(f"checksum mismatch on existing {dest}, redownloading", flush=True)
        dest.unlink()

    tmp = dest.with_suffix(".partial")
    offset = tmp.stat().st_size if tmp.exists() else 0
    if offset > expected_size:
        tmp.unlink()
        offset = 0

    url = f"{BASE_URL}/v2/{REPO}/blobs/{digest}"
    while True:
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urlopen(req, timeout=120) as resp, tmp.open("ab" if offset else "wb") as out:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    offset += len(chunk)
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"retry {digest} at {offset}/{expected_size}: {exc}", flush=True)
            time.sleep(5)

    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {digest}: got {tmp.stat().st_size}, expected {expected_size}")
    actual = sha256_file(tmp)
    if actual != digest:
        raise RuntimeError(f"checksum mismatch for {digest}: got {actual}")
    tmp.replace(dest)
    print(f"DOWNLOADED {digest} {expected_size} bytes", flush=True)


def write_layout(media_type, manifest_digest, manifest_body, manifest, token):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (LAYOUT_DIR / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)

    config = manifest["config"]
    layers = manifest["layers"]
    total = config["size"] + sum(layer["size"] for layer in layers)
    print(f"image: {IMAGE_REF}", flush=True)
    print(f"compressed bytes: {total}", flush=True)
    print(f"layers: {len(layers)}", flush=True)

    manifest_digest_calc = "sha256:" + hashlib.sha256(manifest_body).hexdigest()
    if manifest_digest and manifest_digest != manifest_digest_calc:
        raise RuntimeError(f"manifest digest mismatch: header {manifest_digest}, body {manifest_digest_calc}")
    manifest_digest = manifest_digest_calc

    manifest_dest = blob_path(manifest_digest)
    manifest_dest.write_bytes(manifest_body)

    download_blob(token, config["digest"], config["size"])
    for idx, layer in enumerate(layers, 1):
        print(f"layer {idx}/{len(layers)}", flush=True)
        download_blob(token, layer["digest"], layer["size"])

    (LAYOUT_DIR / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}) + "\n")
    index = {
        "schemaVersion": 2,
        "manifests": [
            {
                "mediaType": media_type,
                "digest": manifest_digest,
                "size": len(manifest_body),
                "annotations": {
                    "org.opencontainers.image.ref.name": TAG,
                    "io.containerd.image.name": IMAGE_REF,
                },
            }
        ],
    }
    (LAYOUT_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def make_archive():
    tmp_archive = ARCHIVE.with_suffix(".oci.tar.partial")
    if tmp_archive.exists():
        tmp_archive.unlink()
    with tarfile.open(tmp_archive, "w") as tar:
        for name in ["oci-layout", "index.json", "blobs"]:
            tar.add(LAYOUT_DIR / name, arcname=name)
    tmp_archive.replace(ARCHIVE)
    print(f"archive: {ARCHIVE}", flush=True)


def main():
    for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(key, None)

    token = get_token()
    media_type, digest, body, manifest = fetch_manifest(token)
    write_layout(media_type, digest, body, manifest, token)
    make_archive()


if __name__ == "__main__":
    main()
