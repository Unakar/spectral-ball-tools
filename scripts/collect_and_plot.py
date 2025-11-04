#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import csv

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None


# Be tolerant to trailing tokens and optional parts order after seed.
# Accept: <method>_n<N>_m<M>_s<seed>[_th<theta>][_pi<pi>][_ms<ms>][_r<rep>]_<anything>.log
LOG_NAME_RE = re.compile(
    r"^(?P<method>[a-z_]+)"                      # method
    r"_n(?P<n>\d+)_m(?P<m>\d+)"                # sizes
    r"_s(?P<seed>\d+)"                          # seed
    r"(?:_th(?P<theta>[^_]+))?"                  # theta (opt)
    r"(?:_pi(?P<pi>\d+))?"                      # power iters (opt)
    r"(?:_ms(?P<ms>\d+))?"                      # msign steps (opt)
    r"(?:_r(?P<rep>\d+))?"                      # repeat index (opt)
    r"(?:_.*)?\.log$"                           # trailing timestamp (opt)
)


def parse_log_file(fp: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "file": fp.name,
        "method": "-",
        "n": "-",
        "m": "-",
        "seed": "-",
        "tol": "-",
        "max_iter": "-",
        "msign_steps": "-",
        "theta_source": "-",
        "power_iters": "-",
        "lambda": "-",
        "abs_f": "-",
        "iters": "-",
        "time_ms": "-",
        "converged": "-",
        "ortho_err": "-",
        "bracket_lo": "-",
        "bracket_hi": "-",
    }

    m = LOG_NAME_RE.match(fp.name)
    if m:
        g = m.groupdict()
        data.update({
            "method": g.get("method") or data["method"],
            "n": g.get("n") or data["n"],
            "m": g.get("m") or data["m"],
            "seed": g.get("seed") or data["seed"],
            "msign_steps": g.get("ms") or data["msign_steps"],
            "theta_source": g.get("theta") or data["theta_source"],
            "power_iters": g.get("pi") or data["power_iters"],
        })

    text = fp.read_text(encoding="utf-8", errors="ignore")

    # header line: method, n,m already handled by filename; tol/max_iter not printed; keep '-'

    # Fallback: parse the run header if present to populate fields
    # Example: [run] method=fixed_point n=4096 m=128 seed=46 rep=5/5 tol=1e-4 ...
    m_hdr = re.search(r"^\[run\]\s+([^\n]+)$", text, re.M)
    if m_hdr:
        hdr = m_hdr.group(1)
        try:
            pairs = re.findall(r"(\w+)=([^\s]+)", hdr)
            kv = dict(pairs)
            data["method"] = kv.get("method", data["method"]) or data["method"]
            data["n"] = kv.get("n", data["n"]) or data["n"]
            data["m"] = kv.get("m", data["m"]) or data["m"]
            data["seed"] = kv.get("seed", data["seed"]) or data["seed"]
            data["tol"] = kv.get("tol", data["tol"]) or data["tol"]
            data["max_iter"] = kv.get("max_iter", data["max_iter"]) or data["max_iter"]
            data["msign_steps"] = kv.get("msign_steps", data["msign_steps"]) or data["msign_steps"]
            # Note: header uses 'theta', map to 'theta_source'
            data["theta_source"] = kv.get("theta", data["theta_source"]) or data["theta_source"]
            data["power_iters"] = kv.get("power_iters", data["power_iters"]) or data["power_iters"]
        except Exception:
            print(f"[warn] failed to parse [run] header in {fp.name}", file=sys.stderr)

    # method from result line: [method]  λ* = ...
    m2 = re.search(r"^\[(?P<m>[^\]]+)\]\s+λ\*\s*=\s*([^\s]+)", text, re.M)
    if m2:
        data["method"] = m2.group("m")
        # lambda value
        lam_match = re.search(r"λ\*\s*=\s*([^\s]+)", m2.group(0))
        if lam_match:
            data["lambda"] = lam_match.group(1)

    # abs f line:  |f(λ*)|     : X  (target ≤ T)
    m3 = re.search(r"\|f\(λ\*\)\|\s*:\s*([0-9.eE+\-]+)", text)
    if m3:
        data["abs_f"] = m3.group(1)

    # iters line: iters       : N iters
    m4 = re.search(r"iters\s*:\s*(\d+)", text)
    if m4:
        data["iters"] = m4.group(1)

    # time line: time : X ms
    m5 = re.search(r"time\s*:\s*([0-9.]+)\s*ms", text)
    if m5:
        data["time_ms"] = m5.group(1)

    # converged line
    m6 = re.search(r"converged\s*:\s*(True|False)", text)
    if m6:
        data["converged"] = m6.group(1)

    # ortho error at λ*
    m7 = re.search(r"orthogonality error @λ\*:\s*([0-9.eE+\-]+)", text)
    if m7:
        data["ortho_err"] = m7.group(1)

    # bracket line optional
    m8 = re.search(r"bracket\s*:\s*\[([^,\]]+),\s*([^\]]+)\]", text)
    if m8:
        data["bracket_lo"] = m8.group(1)
        data["bracket_hi"] = m8.group(2)

    # try to extract tol/max_iter if printed in header footer (optional)
    m_tol = re.search(r"target ≤\s*([0-9.eE+\-]+)", text)
    if m_tol:
        data["tol"] = m_tol.group(1)

    return data


def _format_numeric(value: Any) -> str:
    # Format numerics with compact scientific notation when useful
    try:
        # Keep bool-like strings untouched
        if isinstance(value, str) and value.lower() in ("true", "false", "-"):
            return value
        x = float(value)
        return f"{x:.6g}"
    except Exception:
        return str(value)


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    keys = [
        "file","method","n","m","seed","tol","max_iter","msign_steps",
        "theta_source","power_iters","lambda","abs_f","iters","time_ms",
        "converged","ortho_err","bracket_lo","bracket_hi"
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            # Apply numeric formatting for numeric-looking fields
            formatted = {}
            for k in keys:
                v = r.get(k, "-")
                if k in {"lambda","abs_f","iters","time_ms","ortho_err","bracket_lo","bracket_hi","tol","max_iter"}:
                    formatted[k] = _format_numeric(v)
                else:
                    formatted[k] = v
            w.writerow(formatted)


def average_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group by method, n, m, tol, msign_steps, theta_source, power_iters
    def key(r: Dict[str, Any]) -> Tuple:
        return (
            r.get("method","-"), r.get("n","-"), r.get("m","-"), r.get("tol","-"),
            r.get("msign_steps","-"), r.get("theta_source","-"), r.get("power_iters","-")
        )
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for k, rs in groups.items():
        # 仅平均已收敛的重复（优先用 converged 标志，次选 abs_f ≤ tol）
        sel: List[Dict[str, Any]] = []
        for r in rs:
            ok = str(r.get("converged","-")).lower() == "true"
            if not ok:
                try:
                    absf = float(r.get("abs_f","nan"))
                    tolv = float(r.get("tol","nan")) if r.get("tol") not in (None, "-") else float("nan")
                    ok = (absf <= tolv) if (absf==absf and tolv==tolv) else False
                except Exception:
                    ok = False
            if ok:
                sel.append(r)
        if not sel:
            sel = rs  # 没有收敛样本时，仍保留占位（不阻断绘图）

        def avg_num(field: str) -> str:
            vals: List[float] = []
            for x in sel:
                try:
                    vals.append(float(x.get(field, "nan")))
                except Exception:
                    pass
            return f"{sum(vals)/len(vals):.6g}" if vals else "-"

        anyr = rs[0]
        out.append({
            "file": f"avg-{anyr.get('method','-')}-{anyr.get('n','-')}x{anyr.get('m','-')}",
            "method": anyr.get("method","-"),
            "n": anyr.get("n","-"),
            "m": anyr.get("m","-"),
            "seed": "avg",
            "tol": anyr.get("tol","-"),
            "max_iter": anyr.get("max_iter","-"),
            "msign_steps": anyr.get("msign_steps","-"),
            "theta_source": anyr.get("theta_source","-"),
            "power_iters": anyr.get("power_iters","-"),
            "lambda": avg_num("lambda"),
            "abs_f": avg_num("abs_f"),
            "iters": avg_num("iters"),
            "time_ms": avg_num("time_ms"),
            "converged": "True" if sel and sel is not rs else "-",
            "ortho_err": avg_num("ortho_err"),
            "bracket_lo": "-",
            "bracket_hi": "-",
        })
    return out


def plot_iters_per_shape(avg_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if plt is None:
        print("[warn] matplotlib unavailable; skip plotting", file=sys.stderr)
        return
    # group by (n,m), plot methods vs avg iters
    by_shape: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
    for r in avg_rows:
        by_shape.setdefault((r.get("n","-"), r.get("m","-")), []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    for (n, m), rs in sorted(by_shape.items(), key=lambda x: (int(x[0][0]), int(x[0][1])) if x[0][0].isdigit() and x[0][1].isdigit() else (x[0][0], x[0][1])):
        methods = []
        iters = []
        for r in sorted(rs, key=lambda x: x.get("method","-")):
            methods.append(r.get("method","-"))
            try:
                iters.append(float(r.get("iters","nan")))
            except Exception:
                iters.append(float("nan"))

        plt.figure(figsize=(6,4))
        plt.title(f"Iters to reach tol (n={n}, m={m})")
        plt.bar(methods, iters, color="#4C78A8")
        plt.ylabel("iterations")
        plt.xlabel("method")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"iters_n{n}_m{m}.png"
        plt.savefig(out_path)
        plt.close()


def write_avg_per_shape(avg_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    # Write one CSV per (n,m): averaged metrics per method
    by_shape: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in avg_rows:
        by_shape.setdefault((r.get("n","-"), r.get("m","-")), []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    for (n, m), rows in by_shape.items():
        # Stable ordering by method
        rows = sorted(rows, key=lambda r: r.get("method","-"))
        out = out_dir / f"avg_n{n}_m{m}.csv"
        write_csv(rows, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, default="logs")
    p.add_argument("--out-raw", type=str, default="results/raw.csv")
    p.add_argument("--out-avg", type=str, default="results/avg.csv")
    p.add_argument("--plot-dir", type=str, default="plots")
    p.add_argument("--out-avg-dir", type=str, default="results")
    args = p.parse_args()

    logs_dir = Path(args.logs)
    files = sorted([p for p in logs_dir.glob("*.log")])
    rows: List[Dict[str, Any]] = [parse_log_file(fp) for fp in files]

    write_csv(rows, Path(args.out_raw))
    avg_rows = average_rows(rows)
    write_csv(avg_rows, Path(args.out_avg))
    # Write per-shape averaged CSVs
    write_avg_per_shape(avg_rows, Path(args.out_avg_dir))
    plot_iters_per_shape(avg_rows, Path(args.plot_dir))


if __name__ == "__main__":
    main()
