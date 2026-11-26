# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Apply the reviewed UInt64 UB constructor patch to the pinned Catlass tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PINNED_REVISION = "76946bb3feb6efef4efb204a74fff44196b73c14"
TARGET = "python/tla_dsl/catlass/core_api.py"
PATCH_FILE = "catlass/uint64-ub-tensor.patch"
MANIFEST = Path(__file__).resolve().with_name("catlass-uint64-ub.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git {arguments[0]} failed: {error}")
    return result.stdout


def tracked_changes(root: Path) -> list[str]:
    return sorted(
        name.decode("utf-8")
        for name in git(root, "diff", "--name-only", "-z", "--ignore-submodules=dirty", "HEAD", "--").split(b"\0")
        if name
    )


def quiet_git(root: Path, *arguments: str) -> None:
    if git(root, *arguments).strip():
        raise ValueError(f"unexpected stdout from git {arguments[0]}")


def apply_patch(catlass_root: Path, *, check: bool) -> dict:
    root = catlass_root.resolve(strict=True)
    if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("--catlass-root must name the Catlass Git repository root")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"Catlass revision mismatch: expected {PINNED_REVISION}, got {revision}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dependency") != "catlass"
        or manifest.get("catlass_revision") != PINNED_REVISION
        or manifest.get("target") != TARGET
        or manifest.get("patch_file") != PATCH_FILE
    ):
        raise ValueError("unsupported Catlass patch manifest")
    patch_file = MANIFEST.parent / PATCH_FILE
    patch_hash = digest(patch_file.read_bytes())
    if patch_hash != manifest["patch_sha256"]:
        raise ValueError("Catlass patch SHA256 does not match the reviewed manifest")
    stats = git(root, "apply", "--numstat", "--", str(patch_file)).decode().splitlines()
    if len(stats) != 1 or stats[0].split("\t")[2:] != [TARGET]:
        raise ValueError(f"Catlass patch must change only {TARGET}")

    target = root / TARGET
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError("Catlass patch target must be a regular file within the repository")
    original = git(root, "show", f"{PINNED_REVISION}:{TARGET}")
    if digest(original) != manifest["preimage_sha256"]:
        raise ValueError("pinned Catlass source does not match the reviewed preimage")
    unexpected = [name for name in tracked_changes(root) if name != TARGET]
    if unexpected:
        raise ValueError(f"Catlass has unrelated tracked changes: {unexpected}")
    source_hash = digest(target.read_bytes())
    if source_hash not in (manifest["preimage_sha256"], manifest["postimage_sha256"]):
        raise ValueError("Catlass target source drifted from both reviewed images")

    # Validate the forward result in an isolated tree before changing the caller's
    # source. This also catches an inconsistent postimage entry in the manifest.
    with TemporaryDirectory(prefix="catccos-catlass-patch-") as temporary:
        sandbox = Path(temporary)
        sandbox_target = sandbox / TARGET
        sandbox_target.parent.mkdir(parents=True)
        sandbox_target.write_bytes(original)
        quiet_git(sandbox, "apply", "--check", "--whitespace=error-all", "--", str(patch_file))
        quiet_git(sandbox, "apply", "--whitespace=error-all", "--", str(patch_file))
        if digest(sandbox_target.read_bytes()) != manifest["postimage_sha256"]:
            raise ValueError("Catlass patch result does not match the reviewed postimage")

    if source_hash == manifest["postimage_sha256"]:
        quiet_git(root, "apply", "--reverse", "--check", "--", str(patch_file))
        status = "already_applied"
    else:
        quiet_git(root, "apply", "--check", "--whitespace=error-all", "--", str(patch_file))
        status = "needs_apply"
        if not check:
            quiet_git(root, "apply", "--whitespace=error-all", "--", str(patch_file))
            status = "applied"
            source_hash = digest(target.read_bytes())
            if source_hash != manifest["postimage_sha256"]:
                raise ValueError("Catlass source changed while applying the reviewed patch")
    changes = tracked_changes(root)
    if any(name != TARGET for name in changes):
        raise ValueError(f"Catlass tracked changes drifted during verification: {changes}")
    return {
        "status": status,
        "catlass_revision": revision,
        "patch_sha256": patch_hash,
        "source_sha256": source_hash,
        "expected_sha256": manifest["postimage_sha256"],
        "preimage_sha256": manifest["preimage_sha256"],
        "target": TARGET,
        "tracked_changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catlass-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="verify and report the source state without editing it")
    args = parser.parse_args()
    try:
        result = apply_patch(args.catlass_root, check=args.check)
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
