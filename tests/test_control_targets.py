from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from source_methods import load_class
import torch
import warp as wp

from env.robot_manager.control_manager import MetaControl


class ReadyControl(MetaControl):
    def __init__(self, values):
        super().__init__(values)
        self.calls = []

    def get_action(self, robot_manager, env_idx):
        self.calls.append(env_idx)
        return self.control_info_dict


class ControlTargetsTest(unittest.TestCase):
    def fixture(self, device="cpu"):
        cls = load_class(
            "env/robot_manager/robot_manager.py",
            "RobotManager",
            ["control_robot"],
            dict(np=np, torch=torch, wp=wp, MetaControl=MetaControl),
        )
        obj = cls()
        obj.sim = SimpleNamespace(sim=SimpleNamespace(device=device))
        obj.scene = SimpleNamespace(num_envs=3)
        obj.robot_list = [
            SimpleNamespace(
                robot_type="arm",
                arm_joint_indices=[2, 0],
                gripper_joint_indices=[1, 3],
                arm_name="left",
                gripper_name="left_gripper",
            ),
            SimpleNamespace(
                robot_type="arm",
                arm_joint_indices=[4],
                gripper_joint_indices=[5],
                arm_name="right",
                gripper_name="right_gripper",
            ),
        ]
        obj.process_name = lambda name: name
        calls = []

        def capture(value, joint_ids, env_ids):
            calls.append((value.clone(), joint_ids.numpy().tolist(), env_ids.clone()))

        shared = SimpleNamespace(set_joint_position_target=capture, set_joint_velocity_target=capture)
        obj.robot_key = [shared, shared]
        return obj, calls

    def controls(self, tensor_device=None):
        def tensor_or_list(values):
            if tensor_device is None:
                return values
            return torch.tensor(values, dtype=torch.float64, device=tensor_device, requires_grad=True)

        return [
            ReadyControl(
                {
                    "left": {"position": tensor_or_list([1.1, 2.2]), "velocity": 0.5},
                    "left_gripper": {"position": tensor_or_list([0.2, -0.2])},
                    "right": {"position": [4.0], "velocity": [0.0]},
                    "right_gripper": {"position": 0.4},
                }
            ),
            [],
            ReadyControl(
                {
                    "left": {"position": np.array([3.3, 4.4]), "velocity": [0.7, 0.8]},
                    "left_gripper": {"position": [0.9, -0.9]},
                    "right": {"position": [5.0], "velocity": tensor_or_list([0.1])},
                    "right_gripper": {"position": [0.6]},
                }
            ),
        ]

    def check_targets(self, target_device, tensor_device=None):
        obj, calls = self.fixture(target_device)
        controls = self.controls(tensor_device)
        obj.control_robot(controls)
        expected = [
            [[1.1, 2.2, 0.2, -0.2], [3.3, 4.4, 0.9, -0.9]],
            [[0.5, 0.5], [0.7, 0.8]],
            [[4.0, 0.4], [5.0, 0.6]],
            [[0.0], [0.1]],
        ]
        self.assertEqual(len(calls), len(expected))
        for (value, joints, envs), target in zip(calls, expected):
            self.assertTrue(torch.equal(value, torch.tensor(target, dtype=torch.float32, device=target_device)))
            self.assertFalse(value.requires_grad)
            self.assertEqual(envs.tolist(), [0, 2])
            self.assertEqual(envs.dtype, torch.int32)
            self.assertEqual(value.device, envs.device)
        self.assertEqual([call[1] for call in calls], [[2, 0, 1, 3], [2, 0], [4, 5], [4]])
        self.assertEqual(controls[0].calls, [0])
        self.assertEqual(controls[2].calls, [2])

    def test_host_actions_cpu_targets(self):
        self.check_targets("cpu")

    def test_cpu_tensor_actions(self):
        self.check_targets("cpu", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_host_actions_cuda_targets(self):
        self.check_targets("cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_tensor_actions_stay_off_host(self):
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected host transfer")):
            self.check_targets("cuda:0", "cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_tensor_actions_cpu_targets(self):
        self.check_targets("cpu", "cuda:0")

    def test_empty_batch_does_not_submit_targets(self):
        obj, calls = self.fixture()
        obj.control_robot([[], [], []])
        self.assertEqual(calls, [])

    def test_non_arm_robots_do_not_resolve_actions(self):
        obj, calls = self.fixture()
        for robot in obj.robot_list:
            robot.robot_type = "unsupported"
        controls = self.controls()
        obj.control_robot(controls)
        self.assertEqual(calls, [])
        self.assertEqual(controls[0].calls, [])


class MetaControlMemoTest(unittest.TestCase):
    def fixture(self):
        robot = SimpleNamespace(ee_type="gripper", gripper_scale=[0.0, 1.0], gripper_move={"mimic": [0, 1.0, 0.0]})
        reads = []

        def real_val(robot, env_idx_list):
            reads.append(env_idx_list)
            return {env_idx_list[0]: np.array([0.5])}

        manager = SimpleNamespace(
            _joint_read_cache=None,
            get_robot_obs_name=lambda: ["arm_joint_state", "ee_joint_state"],
            restore_name=lambda name: name,
            get_robot_by_gripper_name=lambda name: robot,
            get_end_effector_real_val=real_val,
        )
        control = MetaControl(
            {"arm_joint_state": {"position": [0.1, 0.2]}, "ee_joint_state": {"position": [0.55, 0.55]}}
        )
        return manager, control, reads

    def test_repeated_calls_share_one_snapshot_result(self):
        manager, control, reads = self.fixture()
        manager._joint_read_cache = {}
        first = control.get_action(manager, env_idx=0)
        self.assertIs(control.get_action(manager, env_idx=0), first)
        self.assertEqual(len(reads), 1)
        # Another env or a new snapshot recomputes.
        control.get_action(manager, env_idx=1)
        manager._joint_read_cache = {}
        control.get_action(manager, env_idx=1)
        self.assertEqual(len(reads), 3)

    def test_no_snapshot_never_memoizes(self):
        manager, control, reads = self.fixture()
        control.get_action(manager, env_idx=0)
        control.get_action(manager, env_idx=0)
        self.assertEqual(len(reads), 2)
