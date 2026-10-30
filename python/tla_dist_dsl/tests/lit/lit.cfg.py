import os
import sys

import lit.formats

# lit injects this instance when it executes the suite configuration.
config = globals()["config"]

config.name = "ascend-catccos-dsl"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir", ".test"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py"]

config.test_source_root = config.tla_dist_lit_source_dir
config.test_exec_root = config.tla_dist_lit_binary_dir

path_entries = []
if getattr(config, "llvm_tools_dir", ""):
    path_entries.append(config.llvm_tools_dir)
for tool in (config.tla_dist_compile_tool, config.filecheck_tool):
    parent = os.path.dirname(tool)
    if parent:
        path_entries.append(parent)
existing_path = os.environ.get("PATH", "")
if existing_path:
    path_entries.append(existing_path)
config.environment["PATH"] = os.pathsep.join(path_entries)

config.substitutions.append(("%tla_dist_compile", config.tla_dist_compile_tool))
config.substitutions.append(("%filecheck", config.filecheck_tool))
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

# Match upstream: Python-generated kernel regressions inherit the native MLIR
# setup and run through the same compiler/FileCheck boundary as handwritten IR.
config.substitutions.append(("%python", sys.executable))
python_paths = [os.path.dirname(os.path.dirname(config.tla_dist_lit_source_dir))]
if os.environ.get("PYTHONPATH"):
    python_paths.append(os.environ["PYTHONPATH"])
config.environment["PYTHONPATH"] = os.pathsep.join(python_paths)
