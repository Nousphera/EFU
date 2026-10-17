"""Scoped compatibility for Flower/Ray environments with spaces in their paths."""

import os
from pathlib import Path
import shlex
import sys


def configure_runtime():
    from flwr.supercore.state.alembic import utils

    original = utils.build_alembic_config

    def build(engine):
        config = original(engine)
        # Flower 1.40 otherwise lets Alembic split each path on spaces.
        locations = [utils.ALEMBIC_DIR / "versions", *utils._version_locations]
        config.set_main_option("path_separator", "os")
        config.set_main_option(
            "version_locations", os.pathsep.join(str(Path(p)) for p in locations)
        )
        return config

    utils.build_alembic_config = build
    import ray

    original_init = ray.init

    def init(*args, **kwargs):
        environment = dict(kwargs.get("runtime_env") or {})
        # Ray 2.55 treats this supported option as a shell command. Quote the
        # interpreter alone; never interpolate command arguments into it.
        environment.setdefault("py_executable", shlex.quote(sys.executable))
        kwargs["runtime_env"] = environment
        return original_init(*args, **kwargs)

    ray.init = init


if __name__ == "__main__":
    configure_runtime()
    from importlib.metadata import entry_points

    name = sys.argv.pop(1)
    entry = tuple(entry_points(group="console_scripts", name=name))[0]
    entry.load()()
