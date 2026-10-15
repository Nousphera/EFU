"""Bound CPU work without changing any tested architecture."""
import torch

def pytest_sessionstart(session):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
