"""Multi-task driver in src/eval_client/main.py, exercised without starting Kit."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_main(task_names, run_task):
    path = ROOT / "src/eval_client/main.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    closed = []
    scope = dict(
        os=os,
        sys=SimpleNamespace(exit=lambda code: (_ for _ in ()).throw(SystemExit(code))),
        TASK_NAMES=task_names,
        MULTI_TASK=len(task_names) > 1,
        run_task=run_task,
        simulation_app=SimpleNamespace(close=lambda: closed.append(True)),
        eval_profiler=SimpleNamespace(reset=lambda: None),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["main"], closed


class MultiTaskMainTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"ROBODOJO_RUN_ID": "base", "ROBODOJO_BASE_RUN_ID": "base"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("ROBODOJO_FAIL_FAST", None)

    def test_each_task_gets_its_own_run_id(self):
        seen = []
        main, closed = load_main(["a", "b"], lambda name: seen.append((name, os.environ["ROBODOJO_RUN_ID"])))
        main()
        self.assertEqual(seen, [("a", "base_a"), ("b", "base_b")])
        self.assertEqual(os.environ["ROBODOJO_RUN_ID"], "base")
        self.assertEqual(closed, [True])

    def test_single_task_keeps_the_plain_run_id(self):
        seen = []
        main, closed = load_main(["a"], lambda name: seen.append(os.environ["ROBODOJO_RUN_ID"]))
        main()
        self.assertEqual(seen, ["base"])
        self.assertEqual(closed, [True])

    def test_failed_task_does_not_stop_the_rest_but_exits_nonzero(self):
        seen = []

        def run_task(name):
            seen.append(name)
            if name == "a":
                raise RuntimeError("boom")

        main, closed = load_main(["a", "b"], run_task)
        with self.assertRaises(SystemExit) as ctx:
            main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual(closed, [True])

    def test_fail_fast_stops_after_first_failure(self):
        os.environ["ROBODOJO_FAIL_FAST"] = "1"
        seen = []

        def run_task(name):
            seen.append(name)
            raise RuntimeError("boom")

        main, _ = load_main(["a", "b"], run_task)
        with self.assertRaises(SystemExit):
            main()
        self.assertEqual(seen, ["a"])


if __name__ == "__main__":
    unittest.main()
