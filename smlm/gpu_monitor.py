"""GPU temperature / power / clock log, one CSV row every `every` seconds (AMD via hwmon or rocm-smi, NVIDIA via
nvidia-smi).

Columns: edge_c (AMD edge / NVIDIA GPU core), junction_c (AMD hotspot; NVIDIA: not reported by nvidia-smi),
mem_c (AMD memory / NVIDIA HBM), power_w, sclk_mhz (shader / SM clock), mclk_mhz, fan (rpm on AMD, % on
NVIDIA, empty for passive cards), vram_used_gib.
"""
import csv
import glob
import json
import os
import re
import subprocess
import time

COLUMNS = ["time", "t_s", "edge_c", "junction_c", "mem_c", "power_w", "sclk_mhz", "mclk_mhz", "fan", "vram_used_gib"]


def _amd_hwmon():
    for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
        try:
            if open(os.path.join(h, "name")).read().strip() == "amdgpu":
                return h
        except OSError:
            pass
    return None


def _rd(p, div=1.0):
    try:
        with open(p) as f:
            return round(int(f.read()) / div, 2)
    except Exception:
        return ""


def _num(s, div=1.0):
    try:
        return round(float(s) / div, 2)
    except ValueError:
        return ""


def _rocm_smi():
    """AMD GPUs without a readable hwmon (e.g. inside a cloud container): rocm-smi --json, keys matched loosely
    because they differ between ROCm versions and GPU generations."""
    try:
        out = subprocess.run(["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--showmeminfo", "vram",
                              "--json"], capture_output=True, text=True, timeout=20).stdout
        d = json.loads(out)
        card = d[sorted(k for k in d if k.startswith("card"))[0]]
    except Exception:
        return None

    def find(*subs):
        for k, v in card.items():
            if all(x in k.lower() for x in subs):
                m = re.search(r"[-+]?\d+(\.\d+)?", str(v))
                if m:
                    return float(m.group(0))
        return ""

    used = find("vram total used")
    return {"edge_c": find("temperature", "edge"), "junction_c": find("temperature", "junction"),
            "mem_c": find("temperature", "memory"), "power_w": find("power"), "sclk_mhz": find("sclk"),
            "mclk_mhz": find("mclk"), "fan": "", "vram_used_gib": round(used / 2**30, 2) if used != "" else ""}


def sample():
    """One reading as a dict of COLUMNS[2:] (empty strings where a sensor is missing)."""
    h = _amd_hwmon()
    if h:
        dev = os.path.dirname(os.path.dirname(h))
        return {"edge_c": _rd(f"{h}/temp1_input", 1000), "junction_c": _rd(f"{h}/temp2_input", 1000),
                "mem_c": _rd(f"{h}/temp3_input", 1000), "power_w": _rd(f"{h}/power1_average", 1e6),
                "sclk_mhz": _rd(f"{h}/freq1_input", 1e6), "mclk_mhz": _rd(f"{h}/freq2_input", 1e6),
                "fan": _rd(f"{h}/fan1_input"), "vram_used_gib": _rd(f"{dev}/mem_info_vram_used", 2**30)}
    r = _rocm_smi()
    if r:
        return r
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,temperature.memory,power.draw,clocks.sm,"
                              "clocks.mem,fan.speed,memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split(", ")
        t, tm, p, sm, mc, fan, mem = (out + [""] * 7)[:7]
        return {"edge_c": _num(t), "junction_c": "", "mem_c": _num(tm), "power_w": _num(p), "sclk_mhz": _num(sm),
                "mclk_mhz": _num(mc), "fan": _num(fan), "vram_used_gib": _num(mem, 1024)}
    except Exception:
        return {k: "" for k in COLUMNS[2:]}


def log_until(path, stop_event, every=10.0):
    t0 = time.time()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        while True:
            w.writerow({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "t_s": round(time.time() - t0, 1), **sample()})
            f.flush()
            if stop_event.wait(every):
                break
