"""Allow the same CLI through ``python -m virassq``."""

from virassq.cli import cli

if __name__ == "__main__":
    cli()
