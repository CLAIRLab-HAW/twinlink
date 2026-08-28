"""Fixtures that reach outside the package: the URDF bundle and the container's SRDF.

``twinlink`` is robot-agnostic, so neither artefact belongs in the package -- the tests locate them through the
workspace marker, the same way the ``_bootstrap`` modules do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "workspace.repos").is_file():
            return parent
    raise RuntimeError("workspace.repos not found above the test file")


@pytest.fixture(scope="session")
def urdf_bundle() -> Path:
    bundle = _workspace_root() / "urdf"
    if not (bundle / "robot.urdf").is_file():
        pytest.skip("urdf/robot.urdf is absent -- run urdf/generate.sh --from-container")
    return bundle


@pytest.fixture(scope="session")
def robot_srdf(tmp_path_factory) -> Path:
    """The SRDF ``move_group`` loads, copied out of the running mock-robot container.

    The oracle must read THIS file: the whole reason for choosing Pinocchio over a second MuJoCo model is that its
    disabled-pair set then agrees with the planner by construction instead of by measurement.
    """
    out = tmp_path_factory.mktemp("srdf") / "robot.srdf"
    copy = subprocess.run(
        ["docker", "cp", "husky-offboard-mock-robot-1:/clearpath/robot.srdf", str(out)],
        capture_output=True,
    )
    if copy.returncode != 0 or not out.is_file():
        pytest.skip("mock-robot container does not answer -- start it with docker compose --profile mock up -d")
    return out
