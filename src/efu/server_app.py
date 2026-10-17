"""Native Flower server application."""

import argparse
import json
from pathlib import Path
import torch
from flwr.serverapp import ServerApp
from .flower import run_server, _experiment

app = ServerApp()


@app.main()
def main(grid, context):
    args = argparse.Namespace(**json.loads(context.run_config["arguments"]))
    path = Path(args.output)
    if path.exists() or path.with_suffix(".pt").exists():
        raise FileExistsError("Choose a new experiment output path")
    experiment = _experiment(args)
    result = run_server(grid, args, experiment=experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    torch.save(
        {
            "model": {
                key: value.detach().cpu()
                for key, value in experiment.model.state_dict().items()
            },
            "configuration": vars(args),
        },
        path.with_suffix(".pt"),
    )
