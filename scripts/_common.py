"""Shared CLI plumbing for scripts/*.py"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", nargs="*", default=None, help="YAML files merged in order (default: configs/*.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="key=value",
                   help="config overrides, e.g. --set data.image_size=224 models.lightpatch.proj_dim=64")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    p.add_argument("--no-pretrained", action="store_true", help="random backbone weights (offline smoke test only)")
    return p


def get_cfg(args):
    cfg = load_config(args.config, args.set)
    if args.device:
        cfg["device"] = args.device
    if args.no_pretrained:
        cfg["pretrained"] = False
    return cfg
