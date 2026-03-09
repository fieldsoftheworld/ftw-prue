import os
import importlib

__all__ = []  # Optional, to control what gets imported with *

# Get the directory of this file
_pkg_dir = os.path.dirname(__file__)

for filename in os.listdir(_pkg_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        modname = filename[:-3]  # strip '.py'
        module = importlib.import_module(f"{__name__}.{modname}")
        globals()[modname] = module  # expose module as subdir1.filenameX
