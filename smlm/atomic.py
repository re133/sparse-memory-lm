"""Write result files in one go: into a temporary file next to the target, then rename. A crash or a killed
process leaves either the old file or none, never half a file that a resumed queue would take as done."""
import json
import os


def _replace(path, write):
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path, obj, **kw):
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, **kw)
    _replace(path, w)


def torch_save(obj, path):
    import torch
    _replace(path, lambda tmp: torch.save(obj, tmp))
