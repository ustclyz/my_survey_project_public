#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_results.py - 汇总平台评测结果 (score_report.json) 为一页表.

用法:
    python tools/summarize_results.py <dir_with_card_subdirs_or_zip>

兼容两种输入: 已解压的目录 (每个卡一个子目录, 内含 score_report.json), 或一个
ZIP (会自动解到临时目录)。
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path


def _load_reports(root: Path):
    out = []
    for p in sorted(root.rglob("score_report.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {p}: {exc}")
            continue
        out.append((p.parent.name, data))
    return out


def _get(d, *path, default=0):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def summarize(entries) -> None:
    rows = []
    for name, d in entries:
        comp = d.get("components") or {}
        counts = d.get("counts") or {}
        rows.append({
            "card": d.get("scenario") or name,
            "total": d.get("total"),
            "sum_best": comp.get("sum_best_scores"),
            "req_pen": comp.get("required_penalty"),
            "req_missing": counts.get("required_missing"),
            "uni_pen": comp.get("uniformity_penalty"),
            "req_reward": comp.get("observation_request_reward"),
            "report_settle": comp.get("report_settlement"),
            "observed": counts.get("targets_observed"),
            "observations": counts.get("observations"),
            "observe_actions": counts.get("observe_actions"),
            "report_issued": counts.get("observation_requests_issued"),
            "report_done": counts.get("observation_requests_completed"),
            "term": _get(d, "termination", "reason", default=""),
        })
    if not rows:
        print("no score_report.json found")
        return
    hdr = f"{'card':<12}{'total':>12}{'sum_best':>11}{'req_miss':>9}{'req_pen':>10}{'uni_pen':>9}{'req_rwd':>9}{'rep_set':>9}{'obs':>7}{'obsp_act':>9}  term"
    print(hdr)
    print("-" * len(hdr))
    tot_A_to_D, n_AD = 0.0, 0
    for r in rows:
        print(f"{str(r['card']):<12}{r['total']:>12.1f}{r['sum_best']:>11.1f}"
              f"{str(r['req_missing']):>9}{r['req_pen']:>10.1f}{r['uni_pen']:>9.1f}"
              f"{r['req_reward']:>9.1f}{r['report_settle']:>9.1f}{str(r['observed']):>7}{str(r['observe_actions']):>9}  {r['term']}")
        sc = str(r["card"])
        if sc in ("v4-a", "v4-b", "v4-c", "v4-d"):
            tot_A_to_D += r["total"] or 0.0
            n_AD += 1
    if n_AD:
        print(f"\nA-D average (overall board) = {tot_A_to_D / n_AD:.2f}   over {n_AD} cards")
    all_tot = sum((r["total"] or 0.0) for r in rows)
    print(f"all-cards sum = {all_tot:.1f} over {len(rows)} cards")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    if target.is_file() and target.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(target) as z:
                z.extractall(td)
            summarize(_load_reports(Path(td)))
    else:
        summarize(_load_reports(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
