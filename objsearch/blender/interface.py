"""Main-venv side of Blender: run objsearch.blender.normalize inside .venv-bpy (separate process: bpy
pins its own numpy and a crash there can't take the dashboard down) and read back its JSON result."""
import json
import os
import subprocess
import sys

from .. import config, track


def bpy_python():
    for p in (os.path.join(config.ROOT, ".venv-bpy", "Scripts", "python.exe"),
              os.path.join(config.ROOT, ".venv-bpy", "bin", "python")):
        if os.path.exists(p):
            return p
    return None


def available():
    return bpy_python() is not None


def normalize(src, out_dir, dims_mm=None, name=None, timeout=900, source_zup=False, render=True,
              srgb_colors=False, pad_finish=None, recolor=None, views=None):
    """-> dict from normalize.py (files, dims_mm, scale_factor, tri_count, flags). Raises on failure."""
    py = bpy_python()
    if py is None:
        raise RuntimeError("Blender isn't installed: create .venv-bpy and `pip install bpy==5.2.*` "
                           "(see objsearch/requirements-bpy.txt)")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [py, "-m", "objsearch.blender.normalize", os.path.abspath(src), os.path.abspath(out_dir)]
    if dims_mm:
        cmd += ["--dims", *[str(d) for d in dims_mm]]
    if name:
        cmd += ["--name", name]
    if source_zup:
        cmd += ["--source-zup"]
    if not render:
        cmd += ["--no-render"]
    if srgb_colors:
        cmd += ["--srgb-colors"]
    if pad_finish:
        cmd += ["--pad-finish", pad_finish]
    if recolor:
        cmd += ["--recolor", json.dumps(recolor)]
    if views:
        cmd += ["--views", ",".join(views)]
    p = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    log_path = os.path.join(out_dir, "blender.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("OBJSEARCH_RESULT ")]
    if p.returncode != 0 or not lines:
        tail = (p.stderr or p.stdout)[-1500:]
        raise RuntimeError(f"Blender normalize failed (exit {p.returncode}); log: {track.rel(log_path)}\n{tail}")
    return json.loads(lines[-1][len("OBJSEARCH_RESULT "):])


def run_module(module, args, log_path, timeout=300):
    """Run python -m <module> <args> in .venv-bpy; -> its OBJSEARCH_RESULT json. Raises on failure."""
    py = bpy_python()
    if py is None:
        raise RuntimeError("Blender venv (.venv-bpy) missing; see objsearch/requirements-bpy.txt")
    p = subprocess.run([py, "-m", module, *args], cwd=config.ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("OBJSEARCH_RESULT ")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"{module} failed (exit {p.returncode}); log: {track.rel(log_path)}\n"
                           f"{(p.stderr or p.stdout)[-1200:]}")
    return json.loads(lines[-1][len("OBJSEARCH_RESULT "):])


if __name__ == "__main__":  # manual: python -m objsearch.blender.interface model.glb out/ 68.6 53.4 15
    print(normalize(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:6]] or None))
