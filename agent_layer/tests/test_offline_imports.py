# coding: utf-8

import os
import subprocess
import sys
import unittest
from pathlib import Path


class OfflineImportTests(unittest.TestCase):
    def test_bootstrap_imports_explicit_dependencies(self):
        from agent_layer.bootstrap import build_application

        self.assertTrue(callable(build_application))

    def test_normalizer_import_has_no_runtime_environment_dependency(self):
        project_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment.pop("LOG_SERVICE_NAME", None)

        result = subprocess.run(
            [sys.executable, "-c", "from agent_layer.ingestion.normalizer import ChunkNormalizer"],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
