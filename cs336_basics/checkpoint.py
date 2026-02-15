import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optim_state_dict": optim_state_dict,
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_state_dict"])
    optimizer.load_state_dict(state_dict["optim_state_dict"])

    return state_dict["iteration"]
