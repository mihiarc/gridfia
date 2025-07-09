"""
Main entry point for the BigMap CLI when run as a module.

This allows running the CLI with `python -m bigmap`.
"""

from bigmap.cli.main import app

if __name__ == "__main__":
    app()