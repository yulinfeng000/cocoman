import sys
from . import maker
from . import uploader
import argparse


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(help="commands")
    maker_parser = sub.add_parser("make", help="make coco dataset")
    uploader_parser = sub.add_parser("upload", help="upload coco dataset")
    maker.get_args(maker_parser)
    maker_parser.set_defaults(func=maker.cmd_entrypoint)
    uploader.get_args(uploader_parser)
    uploader_parser.set_defaults(func=uploader.cmd_entrypoint)
    args = parser.parse_args()
    args.func(args)
