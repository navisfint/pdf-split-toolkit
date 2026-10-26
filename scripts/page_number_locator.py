"""
页码定位器（正式版）
=====================
输入：PDF 路径 + 物理页号
输出：该页底部的页码数字区域（可能含多个数字块），以及裁剪后的小图

算法：
  1. 裁取页面底部区域 (y ∈ [y0f, 1.0])，300dpi 灰度化
  2. 自适应阈值二值化
  3. 连通域分析
  4. 过滤：
       - 排除框线/表格线：w/h > 2.5 或 h/w > 4（细长）
       - 排除过小噪点：h < 页面底部条带高度的 3%
       - 高度需 >= 同排显著水平
  5. 位置约束：横向位于 左 30% 或 右 38% 区域
  6. 在候选里按"高度"降序，取"高度接近最大值的同一行"的所有块 → 组成页码
"""
import pymupdf, numpy as np, os, io
from PIL import Image
from scipy import ndimage


def render_bottom(pdf_path, page_idx, y0f=0.84, dpi=300):
    """裁剪页面底部，返回灰度 np 数组"""
    doc = pymupdf.open(pdf_path)
    try:
        pg = doc[page_idx]
        r = pg.rect
        clip = pymupdf.Rect(r.x0, r.y0 + r.height * y0f, r.x1, r.y1)
        pix = pg.get_pixmap(dpi=dpi, clip=clip)
        arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))
    finally:
        doc.close()
    return arr


def render_full(pdf_path, page_idx, dpi=200):
    """渲染整页（灰度）。dpi 降低以控制内存与速度。"""
    doc = pymupdf.open(pdf_path)
    try:
        pg = doc[page_idx]
        pix = pg.get_pixmap(dpi=dpi)
        arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))
    finally:
        doc.close()
    return arr


def connected_components(arr, thr=None, dark_q=0.02):
    """
    二值化 + 连通域。
    thr 优先使用固定阈值（页码笔画是深黑 30~60，背景 245+，固定 160 最稳）；
    若未指定则用低分位数自适应（2%）。
    """
    H, W = arr.shape
    if thr is None:
        thr = int(np.quantile(arr, dark_q))
        thr = min(max(thr, 120), 200)   # 夹在合理区间，避免阈值过高/过低
    bw = arr < thr
    lbl, n = ndimage.label(bw)
    objs = ndimage.find_objects(lbl)
    comps = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if h <= 0 or w <= 0:
            continue
        comps.append(dict(
            x0=int(xs.start), x1=int(xs.stop),
            y0=int(ys.start), y1=int(ys.stop),
            h=int(h), w=int(w),
            yc=float((ys.start + ys.stop) / 2),
            xc=float((xs.start + xs.stop) / 2),
        ))
    return comps, bw


def is_textlike(w, h):
    """近方形才算文字；细长的是框线/横线"""
    if h <= 0:
        return False
    ar = w / h
    return 0.06 <= ar <= 1.4


def locate_page_number_full(arr,
                            region_top=0.900, region_bot=0.980,
                            x_left=0.20, x_right=0.80,
                            h_min=0.0130, h_max=0.0260,
                            verbose=False):
    """
    页码定位（参数来自 9 个已知样本的实测统计）。

    页码的稳定物理特征：
      - 纵向 yc/H ∈ [0.90, 0.98]      ← 页面最底部一行
      - 高度 h/H ∈ [1.30%, 2.60%]     ← 介于正文小字(1.07%)与表格块(3%+)之间
      - 横向 xc/W < 0.20 (左) 或 > 0.80 (右)
      - 宽高比 w/h ∈ [0.06, 1.4]

    干扰排除：
      - 正文文字  h/H≈1.07%  → 低于 h_min，被排除
      - 表格/签名块 h/H≥3%   → 高于 h_max，被排除
      - 页脚文字带(0.80~0.89) → 低于 region_top，被排除
    """
    H, W = arr.shape
    comps, bw = connected_components(arr)

    text_blocks = [c for c in comps if is_textlike(c['w'], c['h'])]
    if not text_blocks:
        return None

    # 高度窗口 + 纵向带 + 横向边缘 —— 三重硬约束
    cands = []
    for c in text_blocks:
        relh = c['h'] / H
        rely = c['yc'] / H
        relx = c['xc'] / W
        if not (h_min <= relh <= h_max):
            continue
        if not (region_top <= rely <= region_bot):
            continue
        if not (relx < x_left or relx > x_right):
            continue
        c['relh'] = relh; c['rely'] = rely; c['relx'] = relx
        cands.append(c)
    if not cands:
        return None

    # 若多候选：同排聚合成组（多位数页码的各位数字横向相邻）
    cands.sort(key=lambda c: -c['h'])
    best = cands[0]
    max_h = best['h']
    row = [c for c in cands if abs(c['yc'] - best['yc']) <= max_h * 0.6]
    row.sort(key=lambda c: c['xc'])
    groups, cur = [], [row[0]]
    for c in row[1:]:
        if c['xc'] - cur[-1]['xc'] <= max_h * 1.8:
            cur.append(c)
        else:
            groups.append(cur); cur = [c]
    groups.append(cur)
    grp = next((g for g in groups if best in g), [best])

    pad = int(max_h * 0.45)
    x0 = max(0, min(c['x0'] for c in grp) - pad)
    x1 = min(W, max(c['x1'] for c in grp) + pad)
    y0 = max(0, min(c['y0'] for c in grp) - pad)
    y1 = min(H, max(c['y1'] for c in grp) + pad)
    if verbose:
        print(f'  全页 {W}x{H}: 块{len(text_blocks)} 候选{len(cands)} '
              f'组={len(grp)} bbox=({x0},{y0},{x1},{y1}) '
              f'relh={max_h/H:.4f} relx={best["relx"]:.3f} rely={best["rely"]:.4f}')
    return dict(digits=grp, bbox=(x0, y0, x1, y1), max_h=max_h,
                relx=best['relx'], rely=best['rely'],
                n_candidates=len(cands), bw=bw)


def crop_digits(arr, bbox, pad=12):
    """按 bbox 裁出页码小图（带白边，便于 OCR）"""
    x0, y0, x1, y1 = bbox
    H, W = arr.shape
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return arr[y0:y1, x0:x1]
