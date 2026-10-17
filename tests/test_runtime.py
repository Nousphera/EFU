"""Compatibility regressions for installation paths containing spaces."""

import os
import shlex

from alembic.config import Config
from flwr.supercore.state.alembic import utils
import ray
from efu import runtime


def test_runtime_preserves_alembic_paths_and_ray_environment(monkeypatch, tmp_path):
    directories = [
        tmp_path / "install with spaces" / "versions",
        tmp_path / "extra versions",
    ]
    monkeypatch.setattr(utils, "ALEMBIC_DIR", directories[0].parent)
    monkeypatch.setattr(utils, "_version_locations", [directories[1]])
    monkeypatch.setattr(utils, "build_alembic_config", lambda engine: Config())
    monkeypatch.setattr(
        runtime.sys, "executable", str(tmp_path / "python env" / "bin" / "python")
    )
    calls = []
    monkeypatch.setattr(ray, "init", lambda *a, **kw: calls.append(kw))
    runtime.configure_runtime()
    config = utils.build_alembic_config(None)
    assert config.get_main_option("path_separator") == "os"
    assert config.get_main_option("version_locations").split(os.pathsep) == list(
        map(str, directories)
    )
    ray.init(runtime_env={"env_vars": {"OMP_NUM_THREADS": "1"}})
    assert calls[-1]["runtime_env"] == {
        "env_vars": {"OMP_NUM_THREADS": "1"},
        "py_executable": shlex.quote(runtime.sys.executable),
    }
    ray.init(runtime_env={"py_executable": "custom-python"})
    assert calls[-1]["runtime_env"]["py_executable"] == "custom-python"
