"""Compile the actual device route decision with synthetic backend state.

No NPU/topology pass is implied. This guards the zero-base/nonzero-offset
regression even on hosts lacking an actual RoCE-only peer.
"""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_zero_peer_base_cannot_be_hidden_by_nonzero_slot_offset(tmp_path):
    compiler=shutil.which("c++")
    if compiler is None: pytest.skip("host C++ compiler is required for route-policy check")
    header=Path(__file__).resolve().parents[1]/"csrc/mlir/bc/Sync/signal_policy.h"
    source=tmp_path/"policy.cpp"
    source.write_text('''#include "signal_policy.h"
using tla_dist_sync::signalRouteAllowed;
static_assert(signalRouteAllowed(true,true,false,false,false,true));
static_assert(signalRouteAllowed(true,true,true,true,false,false));
static_assert(!signalRouteAllowed(true,false,true,true,true,false));
static_assert(!signalRouteAllowed(false,false,true,true,false,false));
static_assert(signalRouteAllowed(false,false,true,true,true,false));
static_assert(!signalRouteAllowed(false,false,true,true,true,true));
static_assert(!signalRouteAllowed(false,true,true,true,true,true));
static_assert(!signalRouteAllowed(false,true,true,true,false,false));
static_assert(!signalRouteAllowed(false,true,false,true,true,false));
static_assert(!signalRouteAllowed(false,false,true,false,true,false));
static_assert(!signalRouteAllowed(false,false,false,true,true,false));
// Nonzero address arithmetic cannot influence this decision: no slot address
// is accepted as a substitute for a verified topology bit and mapped base.
int main() { return 0; }
''')
    result=subprocess.run([compiler,"-std=c++17","-I",str(header.parent),str(source),"-o",str(tmp_path/"policy")],
        capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    subprocess.run([str(tmp_path/"policy")],check=True)
