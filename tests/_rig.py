"""Test helper: the rig as model-swap sees it, read from the live config.

Tests take card names, ports and paths from here instead of hardcoding one
machine's values. Loading the script runs no command.
"""
import importlib.machinery
import importlib.util
import os

MS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model-swap")


def load():
    loader = importlib.machinery.SourceFileLoader("model_swap", MS)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("model_swap", loader))
    loader.exec_module(mod)
    return mod


ms = load()


def solo_slots():
    """{slot key: gpu name} for every single-card slot off the primary card."""
    out = {}
    for key, sc in (ms.CFG.get("slots") or {}).items():
        if len(sc["gpus"]) == 1 and sc["gpus"][0] != ms.PRIMARY_GPU:
            out[key] = sc["gpus"][0]
    return out
