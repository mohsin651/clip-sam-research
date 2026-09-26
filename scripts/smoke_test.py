import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import _bootstrap
from src.experiment import run
if __name__ == "__main__":
    run(smoke=True)
