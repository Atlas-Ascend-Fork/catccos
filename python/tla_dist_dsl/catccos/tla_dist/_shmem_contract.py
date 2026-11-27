"""Fixed SHMEM capabilities verified against source headers by CMake."""
import json
from pathlib import Path

_CONTRACT = json.loads(Path(__file__).with_suffix(".json").read_text())
SHMEM_MAX_TEAMS = int(_CONTRACT["max_teams"])
SHMEM_REVISION = _CONTRACT["revision"]
