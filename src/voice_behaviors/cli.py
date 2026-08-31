"""Console entrypoint.

`voice-behaviors` with no arguments runs LiveKit's interactive `console` mode
(local mic/speaker). Every LiveKit subcommand still works as usual:
`voice-behaviors dev`, `voice-behaviors start`, `voice-behaviors console --text`.
"""

import logging
import sys

from dotenv import load_dotenv
from livekit.agents import cli as lk_cli

from .worker import create_server

# Shown verbatim rather than being routed into the `console` default.
_HELP_FLAGS = ("--help", "-h")


def _default_to_console(argv: list[str]) -> list[str]:
    """Insert the `console` subcommand when none was given.

    LiveKit's typer app prints its help screen when invoked bare; `make agent` /
    `uv run voice-behaviors` should drop straight into an interactive session.
    All of its options live on the subcommands, so a leading flag (other than
    help) is a console option and gets the same treatment.
    """
    if len(argv) > 1 and (argv[1] in _HELP_FLAGS or not argv[1].startswith("-")):
        return argv
    return [argv[0], "console", *argv[1:]]


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    sys.argv = _default_to_console(sys.argv)
    lk_cli.run_app(create_server())


if __name__ == "__main__":
    main()
