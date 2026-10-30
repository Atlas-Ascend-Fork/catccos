"""Exercise build orchestration without CANN, native compilation, or Git metadata."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_harness(tmp_path):
    project = tmp_path / "project with spaces"
    dsl = project / "python/tla_dist_dsl"
    catlass = project / "3rdparty/catlass/python/tla_dsl"
    shmem = project / "3rdparty/shmem"
    dsl.mkdir(parents=True)
    (catlass / "csrc/mlir").mkdir(parents=True)
    (catlass / "csrc/mlir/CMakeLists.txt").touch()
    (shmem / "include").mkdir(parents=True)
    (shmem / "include/shmem.h").touch()
    shutil.copy2(ROOT / "build.sh", dsl / "build.sh")

    log = tmp_path / "calls.jsonl"
    driver = tmp_path / "record_build.py"
    driver.write_text(
        dedent('''\
            import json
            import os
            import sys
            from pathlib import Path

            role, *args = sys.argv[1:]
            keys = (
                "CATLASS_TLA_DSL_ROOT", "CATLASS_DSL_PREBUILT_ASCENDNPU_IR",
                "CATLASS_DSL_ASCENDNPU_IR_ROOT", "CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR",
                "CMAKE_ARGS", "PIP_NO_BUILD_ISOLATION",
            )
            with open(os.environ["BUILD_TEST_LOG"], "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "role": role, "args": args,
                    "env": {key: os.environ.get(key) for key in keys},
                }) + "\\n")
            if role == "catlass":
                status = int(os.environ.get("BUILD_TEST_CATLASS_EXIT", "0"))
                if status:
                    sys.exit(status)
                location = os.environ.get("BUILD_TEST_BRIDGE_DIR", "catlass")
                if location != "missing":
                    bridge = Path(os.environ["CATLASS_TLA_DSL_ROOT"]) / location
                    bridge.mkdir(parents=True, exist_ok=True)
                    (bridge / "_tla_type_bridge_native.test.so").touch()
            '''),
        encoding="utf-8",
    )
    record_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(driver))}"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scripts = {
        catlass / "build.sh": f'exec {record_command} catlass "$0" "$@"\n',
        fake_bin / "cmake": f'exec {record_command} cmake "$@"\n',
        fake_bin / "python": (
            'if [[ "$*" == "-m pybind11 --cmakedir" ]]; then\n'
            '  printf \'%s\\n\' "$BUILD_TEST_PYBIND11_DIR"\n'
            "else\n"
            f'  exec {shlex.quote(sys.executable)} "$@"\n'
            "fi\n"
        ),
    }
    for path, body in scripts.items():
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
        path.chmod(0o755)

    env = os.environ.copy()
    for key in (
        "CATLASS_DSL_PREBUILT_ASCENDNPU_IR",
        "TLA_DSL_PREBUILT_ASCENDNPU_IR",
        "CATLASS_DSL_ASCENDNPU_IR_ROOT",
        "CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR",
        "CMAKE_ARGS",
        "PIP_NO_BUILD_ISOLATION",
        "BISHENG_COMPILER_PATH",
        "MLIR_TBLGEN_INCLUDE_DIR",
        "BUILD_TEST_CATLASS_EXIT",
        "BUILD_TEST_BRIDGE_DIR",
    ):
        env.pop(key, None)
    build_dir = tmp_path / "build output"
    pybind_dir = tmp_path / "pybind11 cmake"
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
        ASCEND_HOME_PATH=str(tmp_path / "cann"),
        CATLASS_TLA_DSL_ROOT=str(tmp_path / "unrelated catlass"),
        TLA_DIST_DSL_SHMEM_ROOT=str(shmem),
        TLA_DIST_DSL_BUILD_DIR=str(build_dir),
        CMAKE_BUILD_TYPE="Release",
        CC="test-clang",
        CXX="test-clang++",
        BISHENGIR_BUILD_TEMPLATE="ON",
        BUILD_TEST_LOG=str(log),
        BUILD_TEST_PYBIND11_DIR=str(pybind_dir),
    )

    def run():
        result = subprocess.run(
            [str(dsl / "build.sh")], check=False, cwd=tmp_path, env=env, capture_output=True, text=True
        )
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return result, calls

    return SimpleNamespace(
        run=run,
        env=env,
        dsl=dsl,
        catlass=catlass,
        shmem=shmem,
        build_dir=build_dir,
        pybind_dir=pybind_dir,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("bridge_dir", ["catlass", "csrc/mlir/build/python/catlass"])
def test_build_uses_repository_catlass_before_downstream_cmake(build_harness, bridge_dir):
    harness = build_harness
    harness.env["BUILD_TEST_BRIDGE_DIR"] = bridge_dir
    result, calls = harness.run()
    assert result.returncode == 0, result.stderr
    assert [call["role"] for call in calls] == ["catlass", "cmake", "cmake"]
    assert calls[0]["args"] == [str(harness.catlass / "build.sh")]
    assert calls[0]["env"]["CATLASS_TLA_DSL_ROOT"] == str(harness.catlass)
    assert calls[0]["env"]["PIP_NO_BUILD_ISOLATION"] == "false"
    assert shlex.split(calls[0]["env"]["CMAKE_ARGS"]) == [f"-Dpybind11_DIR={harness.pybind_dir}"]
    configure = calls[1]["args"]
    assert configure[configure.index("-S") + 1] == str(harness.dsl / "csrc/mlir")
    assert configure[configure.index("-B") + 1] == str(harness.build_dir)
    assert f"-DCATLASS_TLA_DSL_ROOT={harness.catlass}" in configure
    assert f"-DTLA_DIST_DSL_SHMEM_ROOT={harness.shmem}" in configure
    assert f"-Dpybind11_DIR={harness.pybind_dir}" in configure
    assert calls[2]["args"] == ["--build", str(harness.build_dir), "--target", "tla_dist-compiler"]


@pytest.mark.parametrize(
    "primary,legacy,selected",
    [
        (None, None, None),
        (None, "legacy", "legacy"),
        ("current", None, "current"),
        ("current", "legacy", "current"),
    ],
)
def test_build_shares_ascendnpu_ir_paths_and_preserves_precedence(build_harness, primary, legacy, selected):
    harness = build_harness
    for key, name in (("CATLASS_DSL_PREBUILT_ASCENDNPU_IR", primary), ("TLA_DSL_PREBUILT_ASCENDNPU_IR", legacy)):
        if name is not None:
            harness.env[key] = str(harness.tmp_path / name)
    result, calls = harness.run()
    assert result.returncode == 0, result.stderr
    source = harness.tmp_path / selected if selected else harness.catlass / "3rdparty/AscendNPU-IR"
    for call in calls:
        assert call["env"]["CATLASS_DSL_ASCENDNPU_IR_ROOT"] == str(source)
        assert call["env"]["CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR"] == str(source / "build/install")
    prebuilt_args = [arg for arg in calls[1]["args"] if arg.startswith("-DCATLASS_DSL_PREBUILT_ASCENDNPU_IR=")]
    expected_args = [f"-DCATLASS_DSL_PREBUILT_ASCENDNPU_IR={source}"] if selected else []
    assert prebuilt_args == expected_args


def test_build_preserves_explicit_paths_and_upstream_build_options(build_harness):
    harness = build_harness
    source = str(harness.tmp_path / "explicit source")
    install = str(harness.tmp_path / "explicit install")
    harness.env.update(
        CATLASS_DSL_PREBUILT_ASCENDNPU_IR=str(harness.tmp_path / "prebuilt"),
        CATLASS_DSL_ASCENDNPU_IR_ROOT=source,
        CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR=install,
        CMAKE_ARGS="-DUSER_OPTION=ON",
        PIP_NO_BUILD_ISOLATION="true",
    )
    result, calls = harness.run()
    assert result.returncode == 0, result.stderr
    for call in calls:
        assert call["env"]["CATLASS_DSL_ASCENDNPU_IR_ROOT"] == source
        assert call["env"]["CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR"] == install
    assert calls[0]["env"]["PIP_NO_BUILD_ISOLATION"] == "true"
    assert shlex.split(calls[0]["env"]["CMAKE_ARGS"]) == ["-DUSER_OPTION=ON", f"-Dpybind11_DIR={harness.pybind_dir}"]


@pytest.mark.parametrize("failure", ["upstream_failure", "missing_bridge"])
def test_build_stops_before_cmake_when_catlass_is_not_ready(build_harness, failure):
    harness = build_harness
    if failure == "upstream_failure":
        harness.env["BUILD_TEST_CATLASS_EXIT"] = "23"
    else:
        harness.env["BUILD_TEST_BRIDGE_DIR"] = "missing"
    result, calls = harness.run()
    assert result.returncode == (23 if failure == "upstream_failure" else 1)
    assert [call["role"] for call in calls] == ["catlass"]
    if failure == "missing_bridge":
        assert "without producing _tla_type_bridge_native*.so" in result.stderr
