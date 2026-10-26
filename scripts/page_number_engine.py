"""
全量页码识别 · 工程版
========================
核心改进：不再"单页孤军作战"，而是【全页扫描 + 连续性约束纠错】。

阶段：
  A. 逐页定位页码候选（多阈值尝试，取最可信）
  B. 裁剪小图 → 单字符 OCR
  C. 用"连续递增"约束全局纠错（这是需求难点3的落地）
"""
import pymupdf, numpy as np, os, io, sys, json, traceback
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_number_locator import render_full, connected_components, is_textlike


def locate_candidates(arr, y_lo=0.90, y_hi=0.985, thr_list=(150, 170, 190)):
    """
    多阈值定位页码范围。返回候选块（去重合并）。
    页码物理特征（实测）：y/H ∈ [0.90, 0.985]，h/H ∈ [1.2%, 2.8%]，
                          横向在左右 25% 边缘区
    """
    H, W = arr.shape
    found = []
    for thr in thr_list:
        comps, _ = connected_components(arr, thr=thr)
        for c in comps:
            if not is_textlike(c['w'], c['h']):
                continue
            relh = c['h'] / H
            rely = c['yc'] / H
            relx = c['xc'] / W
            if not (0.012 <= relh <= 0.028):
                continue
            if not (y_lo <= rely <= y_hi):
                continue
            if not (relx < 0.25 or relx > 0.75):
                continue
            found.append(dict(x0=c['x0'], y0=c['y0'], x1=c['x1'], y1=c['y1'],
                              h=c['h'], w=c['w'], xc=c['xc'], yc=c['yc'],
                              relh=relh, rely=rely, relx=relx, thr=thr))
    if not found:
        return None
    # 去重：按 (x0//10, y0//10) 聚类
    found.sort(key=lambda c: -c['h'])
    merged = []
    for c in found:
        dup = False
        for m in merged:
            if abs(m['xc'] - c['xc']) < 20 and abs(m['yc'] - c['yc']) < 20:
                dup = True; break
        if not dup:
            merged.append(c)
    return merged


def group_digits(cands):
    """把同排相邻的数字块聚成页码组"""
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: -c['h'])
    best = cands[0]
    max_h = best['h']
    row = [c for c in cands if abs(c['yc'] - best['yc']) <= max_h * 0.7]
    row.sort(key=lambda c: c['xc'])
    groups, cur = [], [row[0]]
    for c in row[1:]:
        if c['xc'] - cur[-1]['xc'] <= max_h * 2.2:
            cur.append(c)
        else:
            groups.append(cur); cur = [c]
    groups.append(cur)
    return next((g for g in groups if any(x is best for x in g)), [best])
