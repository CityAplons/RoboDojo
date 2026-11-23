"""cuRobo planners are built on first IK use without disturbing RNG streams."""

from copy import deepcopy
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from source_methods import load_class
import torch


class FakePlanner:
    built = 0

    def __init__(self, **kwargs):
        FakePlanner.built += 1
        # Warm-up draws from every global RNG, like cuRobo does.
        random.random()
        np.random.rand()
        torch.rand(1)


def make_manager():
    cls = load_class(
        "env/robot_manager/robot_manager.py",
        "RobotManager",
        ["_setup_planner", "_get_ik_solver", "_build_planner"],
        dict(os=os, random=random, np=np, torch=torch, deepcopy=deepcopy, CuroboPlanner=FakePlanner),
    )
    obj = cls()
    obj.planner, obj.ik_solver, obj._planner_robots, obj.dt = {}, {}, {}, 0.004
    return obj


ROBOT = SimpleNamespace(
    robot_type="arm",
    robot_name="arm0",
    entity_origin_pose=[0, 0, 0.1],
    arm_joints_name=["j1"],
    curobo_yml_path="x.yml",
)


class LazyPlannerTest(unittest.TestCase):
    def setUp(self):
        FakePlanner.built = 0
        os.environ.pop("ROBODOJO_EAGER_PLANNER", None)

    def test_built_once_on_first_use(self):
        obj = make_manager()
        obj._setup_planner(ROBOT)
        self.assertEqual(FakePlanner.built, 0)
        first = obj._get_ik_solver(ROBOT)
        self.assertIs(obj._get_ik_solver(ROBOT), first)
        self.assertEqual(FakePlanner.built, 1)

    def test_build_leaves_rng_streams_untouched(self):
        obj = make_manager()
        obj._setup_planner(ROBOT)
        random.seed(3), np.random.seed(3), torch.manual_seed(3)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        random.seed(3), np.random.seed(3), torch.manual_seed(3)
        obj._get_ik_solver(ROBOT)
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected)

    def test_eager_opt_in_builds_immediately(self):
        obj = make_manager()
        with patch.dict(os.environ, {"ROBODOJO_EAGER_PLANNER": "1"}):
            obj._setup_planner(ROBOT)
        self.assertEqual(FakePlanner.built, 1)


if __name__ == "__main__":
    unittest.main()
