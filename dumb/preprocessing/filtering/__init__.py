import os
import glob

# Get all Python files in this folder (excluding __init__.py)
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.endswith("__init__.py")]

# Import all modules dynamically
for module in __all__:
    exec(f"from .{module} import *")
