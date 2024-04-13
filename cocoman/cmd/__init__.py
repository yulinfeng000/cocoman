import sys
from . import maker
from . import uploader


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'make':
            args = maker.get_args()
            maker.cmd_entrypoint(args)
        elif sys.argv[1] == 'upload':
            args = uploader.get_args()
            uploader.cmd_entrypoint(args)
        else:
            print("Unknown command. Available commands: upload, make")
    else:
        print("No command provided. Available commands: upload, make")