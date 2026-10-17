"""Run the source tree with Flower's native simulation runtime."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import socket
import time
import urllib.request
import sys
from tempfile import TemporaryDirectory
import tomli_w


def parser():
    from .train import build_parser

    p = build_parser()
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--client-cpus", type=int, default=1)
    p.add_argument("--client-gpus", type=float, default=None)
    p.add_argument("--parallel-clients", type=int, default=1)
    p.add_argument("--registration-timeout", type=float, default=120.0)
    p.add_argument("--reply-timeout", type=float, default=3600.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without loading data or launching clients",
    )
    return p


def configuration(args):
    from .train import validate_args

    if args.parallel_clients < 1 or args.client_cpus < 1 or args.torch_threads < 1:
        raise ValueError("CPU resources and parallel client count must be positive")
    if not 0 < args.registration_timeout <= 3600 or not 0 < args.reply_timeout <= 86400:
        raise ValueError("Timeouts must be positive and bounded")
    if args.client_gpus is None:
        args.client_gpus = 1.0 if args.device.startswith("cuda") else 0.0
    if not math.isfinite(args.client_gpus):
        raise ValueError("GPU resources must be finite")
    if args.client_gpus < 0 or (
        args.device.startswith("cuda") and args.client_gpus <= 0
    ):
        raise ValueError("CUDA clients require positive GPU resources")
    for key in ("data_root", "model_root"):
        if hasattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    if not args.output:
        method = args.method
        if __package__ == "efu" and method == "efu":
            method += "-" + args.unlearn_method + "-" + args.variant
        args.output = (
            f"results/{method}-{args.model}-{args.dataset}-seed{args.seed}.json"
        )
    args.output = str(Path(args.output).expanduser().resolve())
    validate_args(args)
    runtime_log = Path(args.output).with_suffix(".runtime.log")
    if runtime_log.exists():
        raise FileExistsError(
            f"Runtime log already exists: {runtime_log}; choose a new output name"
        )
    return args


@contextmanager
def flower_app(args):
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    name = __package__
    descriptor = {
        "build-system": {
            "requires": ["setuptools>=77", "wheel"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": name,
            "version": "1.0.0",
            "description": name + " federated research",
            "requires-python": ">=3.11,<3.14",
            "license": "Apache-2.0 AND MIT",
            "dependencies": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "nousphera",
                    "components": {
                        "serverapp": name + ".server_app:app",
                        "clientapp": name + ".client_app:app",
                    },
                    "config": {"arguments": json.dumps(vars(args), allow_nan=False)},
                }
            }
        },
    }
    with TemporaryDirectory(prefix=name + "-flower-") as directory:
        target = Path(directory)
        shutil.copytree(
            package,
            target / "src" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
        (target / "setup.py").write_text(
            'from setuptools import setup,find_packages\nsetup(packages=find_packages("src"),package_dir={"":"src"})\n'
        )
        shutil.copy2(root / "LICENSE", target / "LICENSE")
        (target / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield target


@contextmanager
def local_runtime(args, env):
    """Own an isolated loopback runtime, preserving unrelated Flower history."""
    with TemporaryDirectory(prefix=__package__ + "-runtime-") as directory:
        root = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        commands = root / "bin"
        commands.mkdir()
        for name in (
            "flwr",
            "flower-superexec",
            "flwr-simulation",
            "flwr-serverapp",
            "flwr-clientapp",
        ):
            wrapper = commands / name
            command = shlex.join([sys.executable, "-m", __package__ + ".runtime", name])
            wrapper.write_text("#!/bin/sh\nexec " + command + ' "$@"\n')
            wrapper.chmod(0o755)
        runtime_env = env | {
            "FLWR_HOME": str(root),
            "PATH": str(commands) + os.pathsep + env.get("PATH", ""),
        }
        (root / "config.toml").write_text(
            tomli_w.dumps(
                {
                    "superlink": {
                        "default": "experiment",
                        "experiment": {
                            "address": f"127.0.0.1:{port}",
                            "insecure": True,
                        },
                    }
                }
            )
        )
        log_path = Path(args.output).with_suffix(".runtime.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    __package__ + ".runtime",
                    "flower-superlink",
                    "--simulation",
                    "--insecure",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--database",
                    str(root / "state.db"),
                    "--disable-runtime-dependency-installation",
                ],
                env=runtime_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + args.registration_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Flower runtime stopped; inspect {log_path}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=0.5
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Flower runtime did not start; inspect {log_path}"
                        )
                    time.sleep(0.1)
                yield runtime_env
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        configuration(args)
        if args.dry_run:
            print(json.dumps(vars(args), indent=2))
            return
        from .experiment import Experiment

        Experiment(args)  # fail on missing data/weights before starting services
        binary = Path(sys.executable).with_name("flwr")
        if not binary.is_file():
            raise RuntimeError("Install requirements.txt in the active environment")
        resources = (
            f"num-supernodes={args.clients} client-resources-num-cpus={args.client_cpus} "
            f"client-resources-num-gpus={args.client_gpus} init-args-num-cpus={args.client_cpus * args.parallel_clients}"
        )
        env = os.environ.copy()
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
        with local_runtime(args, env) as runtime_env, flower_app(args) as app:
            subprocess.run(
                [
                    shutil.which("flwr", path=runtime_env["PATH"]),
                    "run",
                    str(app),
                    "experiment",
                    "--federation-config",
                    resources,
                    "--stream",
                ],
                check=True,
                env=runtime_env,
            )
        path = Path(args.output)
        if not path.is_file() or not path.with_suffix(".pt").is_file():
            raise RuntimeError("Flower finished without results; inspect the run log")
        print(f"Saved {path}")
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
