#app/components/path_utils.py

import os

def get_project_root():
    # Returns path to `src/..` → the root of your project
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))