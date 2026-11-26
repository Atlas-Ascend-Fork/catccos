# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify patch identity, idempotency and drift rejection in disposable trees."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "tools/patches"
SCRIPT = PATCHES / "apply_catlass_patches.py"
MANIFEST = json.loads((PATCHES / "catlass-uint64-ub.json").read_text())


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def catlass_tree(tmp_path):
    source = ROOT.parents[1] / "3rdparty/catlass"
    if not (source / "python/tla_dsl/catlass/core_api.py").is_file():
        pytest.skip("initialize the pinned Catlass submodule for patch tests")
    target = tmp_path / "catlass source"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    git(target, "sparse-checkout", "set", "--no-cone", "/" + MANIFEST["target"])
    git(target, "checkout", "--detach", MANIFEST["catlass_revision"])
    return target


def run_patch(root, *args, script=SCRIPT):
    return subprocess.run(
        [sys.executable, str(script), "--catlass-root", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def source_hash(root):
    return hashlib.sha256((root / MANIFEST["target"]).read_bytes()).hexdigest()


def test_patch_check_apply_and_reapply_are_exact_and_idempotent(catlass_tree):
    checked = run_patch(catlass_tree, "--check")
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == "needs_apply"
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]
    applied = run_patch(catlass_tree)
    assert applied.returncode == 0, applied.stderr
    receipt = json.loads(applied.stdout)
    assert receipt["status"] == "applied"
    assert receipt["catlass_revision"] == MANIFEST["catlass_revision"]
    assert receipt["source_sha256"] == receipt["expected_sha256"] == MANIFEST["postimage_sha256"]
    assert receipt["patch_sha256"] == MANIFEST["patch_sha256"]
    assert receipt["tracked_changes"] == [MANIFEST["target"]]
    for args in ((), ("--check",)):
        repeated = run_patch(catlass_tree, *args)
        assert repeated.returncode == 0, repeated.stderr
        assert json.loads(repeated.stdout)["status"] == "already_applied"
        assert source_hash(catlass_tree) == MANIFEST["postimage_sha256"]


def test_patch_rejects_source_drift(catlass_tree):
    target = catlass_tree / MANIFEST["target"]
    target.write_bytes(target.read_bytes() + b"\n# unrelated local change\n")
    before = source_hash(catlass_tree)
    result = run_patch(catlass_tree)
    assert result.returncode == 1
    assert "drifted from both reviewed images" in result.stderr
    assert source_hash(catlass_tree) == before


def test_patch_rejects_other_tracked_changes(catlass_tree):
    other = "python/tla_dsl/catlass/types.py"
    git(catlass_tree, "sparse-checkout", "add", "/" + other)
    target = catlass_tree / other
    target.write_bytes(target.read_bytes() + b"\n# unrelated local change\n")
    result = run_patch(catlass_tree)
    assert result.returncode == 1
    assert "unrelated tracked changes" in result.stderr
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]


def test_patch_rejects_wrong_catlass_revision(catlass_tree):
    parent = git(catlass_tree, "rev-parse", MANIFEST["catlass_revision"] + "^")
    git(catlass_tree, "update-ref", "HEAD", parent)
    result = run_patch(catlass_tree)
    assert result.returncode == 1
    assert "Catlass revision mismatch" in result.stderr
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]


@pytest.fixture
def patch_copy(tmp_path):
    destination = tmp_path / "patch tools"
    shutil.copytree(PATCHES, destination)
    return destination


def test_patch_rejects_modified_patch_bytes(catlass_tree, patch_copy):
    patch = patch_copy / MANIFEST["patch_file"]
    patch.write_bytes(patch.read_bytes() + b"\n")
    result = run_patch(catlass_tree, script=patch_copy / SCRIPT.name)
    assert result.returncode == 1
    assert "patch SHA256" in result.stderr
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]


def test_patch_rejects_additional_target_even_with_updated_patch_hash(catlass_tree, patch_copy):
    patch = patch_copy / MANIFEST["patch_file"]
    patch.write_bytes(
        patch.read_bytes()
        + b"diff --git a/extra.txt b/extra.txt\nnew file mode 100644\n--- /dev/null\n+++ b/extra.txt\n@@ -0,0 +1 @@\n+unexpected\n"
    )
    manifest = dict(MANIFEST, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    (patch_copy / "catlass-uint64-ub.json").write_text(json.dumps(manifest))
    result = run_patch(catlass_tree, script=patch_copy / SCRIPT.name)
    assert result.returncode == 1
    assert "must change only" in result.stderr
    assert not (catlass_tree / "extra.txt").exists()
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]


def test_patch_rejects_inconsistent_postimage_before_editing_source(catlass_tree, patch_copy):
    manifest = dict(MANIFEST, postimage_sha256="0" * 64)
    (patch_copy / "catlass-uint64-ub.json").write_text(json.dumps(manifest))
    result = run_patch(catlass_tree, script=patch_copy / SCRIPT.name)
    assert result.returncode == 1
    assert "does not match the reviewed postimage" in result.stderr
    assert source_hash(catlass_tree) == MANIFEST["preimage_sha256"]
