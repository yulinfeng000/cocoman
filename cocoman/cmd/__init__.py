from cocoman.cmd import uploader
import argparse


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(help="commands")
    uploader_parser = sub.add_parser("upload", help="upload coco dataset")
    uploader.get_args(uploader_parser)
    uploader_parser.set_defaults(func=uploader.cmd_entrypoint)
    args = parser.parse_args()
    args.func(args)
