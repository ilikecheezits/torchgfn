import importlib.metadata as met

try:
    __version__ = met.version("torchgfn")
except met.PackageNotFoundError:
    __version__ = "0.0.0-dev"
