try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("heatcluster")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

from .cli import main
