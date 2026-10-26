"""
页码映射引擎（正式版）
======================
核心算法：分段偏移法
  页码 = 物理页 − offset，其中 offset 分段恒定

步骤：
  1. 抽样 OCR（每 sample_step 页取 1 页）
  2. 多数表决求主导 offset，剔除离群样本（OCR 错误）
  3. 滑窗检测 offset 跳变点 → 分段
  4. 段内全量推算页码
  5. 输出 物理页↔页码 完整映射 + 异常清单
"""
import pymupdf, numpy as np, os, io, re, json
from PIL import Image
from collections import Counter
from scipy import ndimage

# ---------- 页码定位 ----------

def is_textlike(w, h):
    if h <= 0:
        return False
    return 0.06 <= (w / h) <= 1.4


def render_full(pdf_path, page_idx, dpi=200):
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_idx].get_pixmap(dpi=dpi)
        arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))
    finally:
        doc.close()
    return arr


def connected_components(arr, thr=160):
    bw = arr < thr
    lbl, n = ndimage.label(bw)
    objs = ndimage.find_objects(lbl)
    comps = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start; w = xs.stop - xs.start
        if h <= 0 or w <= 0:
            continue
        comps.append(dict(x0=int(xs.start), x1=int(xs.stop),
                          y0=int(ys.start), y1=int(ys.stop),
                          h=int(h), w=int(w),
                          yc=float((ys.start+ys.stop)/2),
                          xc=float((xs.start+xs.stop)/2)))
    return comps, bw


def locate_page_number(arr, y_lo=0.885, y_hi=0.995,
                       x_edge=(0.145, 0.855)):
    """
    定位页码。返回 bbox + 元信息，或 None。

    参数依据（实测统计）：
      - rely ∈ [0.885, 0.995]  页面最底部一行
      - relx < 0.145 或 > 0.855  【贴边】——这是区分页码与页脚文字的关键
           页码贴页边(relx≈0.05~0.13 / 0.90~0.97)
           页脚文字有缩进(relx≈0.20~0.30 / 0.72~0.83)
      - relh ∈ [0.0075, 0.030]  字号区间
      - 宽高比 0.06~1.4（数字竖长）

    多阈值投票：不同阈值的二值化结果合并，提升召回。
    """
    H, W = arr.shape
    lt, rt = x_edge
    cands = []
    for thr in (130, 150, 170, 190, 215):
        comps, _ = connected_components(arr, thr=thr)
        for c in comps:
            if not is_textlike(c['w'], c['h']):
                continue
            relh = c['h'] / H; rely = c['yc'] / H; relx = c['xc'] / W
            if not (0.0075 <= relh <= 0.030):
                continue
            if not (y_lo <= rely <= y_hi):
                continue
            if not (relx < lt or relx > rt):
                continue
            cands.append((relh, rely, relx, c, thr))
    if not cands:
        return None

    # 排序：越靠页面底部、越贴边 → 越可能是页码；同分取字号大
    def score(item):
        relh, rely, relx, c, thr = item
        edge = min(relx, 1 - relx)
        return (rely, -edge, -relh)
    cands.sort(key=score, reverse=True)

    # 取最靠底部的一簇
    best = cands[0]
    max_h = best[3]['h']
    row = [c for _,_,_,c,_ in cands if abs(c['yc']-best[3]['yc']) <= max_h*0.8]
    row.sort(key=lambda c: c['xc'])
    groups, cur = [], [row[0]]
    for c in row[1:]:
        if c['xc'] - cur[-1]['xc'] <= max_h*2.5:
            cur.append(c)
        else:
            groups.append(cur); cur = [c]
    groups.append(cur)
    grp = next((g for g in groups if any(x is best[3] for x in g)), [best[3]])

    pad = int(max_h*0.55)
    x0 = max(0, min(c['x0'] for c in grp)-pad)
    x1 = min(W, max(c['x1'] for c in grp)+pad)
    y0 = max(0, min(c['y0'] for c in grp)-pad)
    y1 = min(H, max(c['y1'] for c in grp)+pad)
    return dict(bbox=(x0,y0,x1,y1), n_digits=len(grp),
                relx=best[2], rely=best[1], relh=best[0])


def crop_page_number(arr, bbox, scale=3):
    x0,y0,x1,y1 = bbox
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    im = Image.fromarray(crop)
    im = im.resize((im.width*scale, im.height*scale), Image.LANCZOS)
    return im


# ---------- 页码映射 ----------

def _as_int(s):
    if s is None:
        return None
    s = str(s).strip()
    return int(s) if re.fullmatch(r'\d{1,4}', s) else None


def build_page_map(observations, total_pages, front_offset_guess=None,
                   verbose=False):
    """
    由抽样观测构建完整页码映射。

    observations: list of dict(page=物理页1based, value=页码int或None)
    返回 dict:
      segments: [{start, end, offset, page_range}]
      page_map: {物理页: 页码}
      anomalies: [...]
    """
    obs = [(o['page'], o['value']) for o in observations
           if o['value'] is not None]
    if not obs:
        return dict(segments=[], page_map={}, anomalies=[
            dict(type='NO_OBSERVATION', detail='没有任何有效页码观测')])

    # --- 1) 求主导 offset（多数表决）---
    offs = Counter(p - v for p, v in obs)
    dominant_off, dominant_cnt = offs.most_common(1)[0]
    if verbose:
        print(f'主导 offset={dominant_off} ({dominant_cnt}/{len(obs)})')

    # --- 2) 保留与主导 offset 一致的样本（剔除 OCR 错误）---
    clean = [(p, v) for p, v in obs if (p - v) == dominant_off]
    anomalies = []
    outliers = [(p, v) for p, v in obs if (p - v) != dominant_off]
    for p, v in outliers:
        anomalies.append(dict(type='OCR_OUTLIER', page=p, value=v,
                              implied_offset=p - v))

    # --- 3) 分段：滑窗检测 offset 跳变 ---
    clean.sort()
    segments = []
    seg_start = clean[0][0]
    seg_off = clean[0][0] - clean[0][1]
    prev_page = clean[0][0]
    for p, v in clean[1:]:
        off = p - v
        if off != seg_off:
            # 记录上一段
            segments.append(dict(start_phy=seg_start, end_phy=prev_page,
                                 offset=seg_off,
                                 start_page=seg_start - seg_off,
                                 end_page=prev_page - seg_off))
            anomalies.append(dict(type='OFFSET_SHIFT', at_page=prev_page+1,
                                 prev_offset=seg_off, new_offset=off,
                                 inserted=(off - seg_off)))
            seg_start = p; seg_off = off
        prev_page = p
    segments.append(dict(start_phy=seg_start, end_phy=prev_page,
                         offset=seg_off,
                         start_page=seg_start - seg_off,
                         end_page=prev_page - seg_off))

    # --- 4) 段间空缺填补：把 segment 扩展到连续覆盖 ---
    if segments:
        # 第一段向前扩展到正文起点（用其 offset 推算）
        segments[0]['start_phy'] = max(1, segments[0]['start_phy'])
        segments[-1]['end_phy'] = total_pages
        # 段间缝隙：用前一段 offset 延续
        for i in range(len(segments)-1):
            segments[i]['end_phy'] = max(segments[i]['end_phy'],
                                         segments[i+1]['start_phy'] - 1)

    # --- 5) 全量推算 ---
    page_map = {}
    for s in segments:
        for phy in range(s['start_phy'], s['end_phy']+1):
            page_map[phy] = phy - s['offset']

    return dict(segments=segments, page_map=page_map,
                dominant_offset=dominant_off, anomalies=anomalies,
                n_obs=len(obs), n_clean=len(clean))
