"""
卷内目录解析器
==============
从目录页 OCR 结果重建表格行，提取 (序号, 文件题名, 起始页码)。

关键点：
  1. OCR 返回的文本顺序 ≠ 阅读顺序 → 必须用 bbox 的 y 坐标聚类成"行"
  2. 页码列在最右侧，通过 x 坐标定位
  3. 支持两种页码格式：单个数字 "7" / 区间 "194~196"
  4. 表头行(序号/文件编号/责任者/...)与数据行要区分
"""
import re
import numpy as np
from collections import defaultdict


def _bbox_of(poly_or_box):
    """把 OCR 的框统一成 (x0,y0,x1,y1)"""
    b = np.array(poly_or_box, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 2)
    return (float(b[:, 0].min()), float(b[:, 1].min()),
            float(b[:, 0].max()), float(b[:, 1].max()))


def ocr_items(result):
    """把 PaddleOCR predict 结果拍平成 [(text, score, x0,y0,x1,y1), ...]"""
    items = []
    for r in result:
        texts = r.get('rec_texts', [])
        scores = r.get('rec_scores', [])
        boxes = r.get('rec_boxes', r.get('dt_polys', []))
        for t, s, b in zip(texts, scores, boxes):
            if b is None or len(b) == 0:
                continue
            x0, y0, x1, y1 = _bbox_of(b)
            items.append(dict(text=str(t).strip(), score=float(s),
                              x0=x0, y0=y0, x1=x1, y1=y1,
                              xc=(x0+x1)/2, yc=(y0+y1)/2))
    return items


def cluster_rows(items, y_tol_ratio=0.008, page_h=None):
    """按 y 坐标把文本聚类成行"""
    if not items:
        return []
    if page_h is None:
        page_h = max(i['y1'] for i in items) or 1.0
    tol = page_h * y_tol_ratio if page_h > 2 else 12
    items = sorted(items, key=lambda i: i['yc'])
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it['yc'] - cur[-1]['yc']) <= tol:
            cur.append(it)
        else:
            rows.append(cur); cur = [it]
    rows.append(cur)
    for r in rows:
        r.sort(key=lambda i: i['x0'])
    return rows


# ---- 页码列解析 ----

PAGE_RANGE_RE = re.compile(r'^\s*(\d{1,4})\s*[~～\-—至]\s*(\d{1,4})\s*$')
PAGE_SINGLE_RE = re.compile(r'^\s*(\d{1,4})\s*$')
# OCR 常见混淆：末位数字被识别成字母
DIGIT_FIX = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1',
                           'S': '5', 's': '5', 'B': '8', 'g': '9',
                           'Z': '2', 'z': '2', 'q': '9'})


def parse_page_cell(text):
    """
    解析页码单元格。返回 (start, end) 或 None。
    支持: "7" / "194~196" / "166～182" / "117~128"
    """
    if text is None:
        return None
    t = str(text).strip()
    t = t.replace('，', '~').replace(',', '~').replace('、', '~')
    m = PAGE_RANGE_RE.match(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = PAGE_SINGLE_RE.match(t)
    if m:
        return (int(m.group(1)), int(m.group(1)))
    # 尝试修字
    t2 = t.translate(DIGIT_FIX)
    m = PAGE_RANGE_RE.match(t2)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = PAGE_SINGLE_RE.match(t2)
    if m:
        return (int(m.group(1)), int(m.group(1)))
    return None


def extract_toc_rows(rows, page_w, page_h=None, header_keywords=None,
                     idx_col_max=0.16):
    """
    从聚类的行里提取目录条目。

    锚点法（比纯行聚类稳健）：
      1. 先找出所有"序号单元格"（最左列的纯数字）作为行锚点
      2. 每个锚点确定一个表格条目的纵向区间（到下一个锚点为止）
      3. 区间内的所有单元格归属该条目
      4. 页码取区间内最右侧的可解析数字
    这样即使题名换行、单元格 y 不齐，也不会错行。

    idx_col_max: 序号列 x 归一化上限。实测各卷宗 0.095~0.145，默认 0.16。
                 注意必须 < 页码列位置(0.72)，且不能太宽以免把责任者数字吞进来。
    """
    if header_keywords is None:
        header_keywords = ['序号', '文件编号', '责任者', '文件题名', '文件题目',
                           '日期', '页号', '页码', '备注']

    # 拍平所有单元格
    cells = [it for r in rows for it in r]
    if not cells:
        return []

    # 过滤表头/标题单元格
    def is_header(c):
        t = c['text']
        return any(k in t for k in header_keywords) or '卷内目录' in t or '档号' in t

    body = [c for c in cells if not is_header(c)]
    if not body:
        return []

    # 1) 找序号锚点（序号列实测 xc/W ≈ 0.095~0.145）
    anchors = []
    for c in body:
        if c['xc'] / page_w < idx_col_max:
            t = c['text'].translate(DIGIT_FIX).strip()
            m = re.fullmatch(r'(\d{1,3})', t)
            if m:
                anchors.append(dict(idx=int(m.group(1)), yc=c['yc'],
                                    x0=c['x0'], y0=c['y0'], y1=c['y1']))
    if not anchors:
        return []
    anchors.sort(key=lambda a: a['yc'])

    # 去重（同一序号可能被多阈值识别）
    dedup = []
    for a in anchors:
        if dedup and a['idx'] == dedup[-1]['idx'] and abs(a['yc']-dedup[-1]['yc']) < 30:
            continue
        dedup.append(a)
    anchors = dedup

    # 2) 为每个锚点划定纵向区间
    n = len(anchors)
    out = []
    for i, a in enumerate(anchors):
        if i > 0 and i < n-1:
            y_lo = (anchors[i-1]['yc'] + a['yc']) / 2
            y_hi = (a['yc'] + anchors[i+1]['yc']) / 2
        elif i == 0 and n > 1:
            y_lo = a['yc'] - (anchors[1]['yc'] - a['yc'])
            y_hi = (a['yc'] + anchors[1]['yc']) / 2
        elif i == n-1 and n > 1:
            y_lo = (anchors[i-1]['yc'] + a['yc']) / 2
            y_hi = a['yc'] + (a['yc'] - anchors[i-1]['yc'])
        else:
            # 单锚点页：整页作为其区间
            y_lo = 0
            y_hi = float('inf')
        region = [c for c in body if y_lo <= c['yc'] <= y_hi]

        # 页码：取区域内【最右侧那一列】。自适应列定位，避免误吞日期列。
        #   实测：日期列 xc/W≈0.75~0.79，页码列 xc/W≈0.83~0.88 —— 靠阈值难分，
        #   改用"以最右 cell 为锚，取 x 邻近者"，稳健得多。
        digit_cells = [c for c in region
                       if re.search(r'\d', c['text'])]
        page = None
        page_cell = None
        if digit_cells:
            x_anchor = max(c['xc'] for c in digit_cells)
            col_tol = page_w * 0.06
            col = [c for c in digit_cells if abs(c['xc'] - x_anchor) <= col_tol]
            col.sort(key=lambda c: c['yc'])
            joined = ''.join(c['text'] for c in col)
            pv = parse_page_cell(joined)
            if pv:
                page = pv
                page_cell = joined
            else:
                # 退而求其次：单 cell 逐个试，优先长的
                for c in sorted(col, key=lambda c: (-len(c['text']), c['yc'])):
                    pv = parse_page_cell(c['text'])
                    if pv:
                        page = pv; page_cell = c['text']; break

        # 题名：中间区域最长的文本（排除序号列与页码列）
        cands = [c for c in region
                 if 0.10 < c['xc']/page_w < 0.72
                 and not re.fullmatch(r'[\d\.\s]+', c['text'])]
        title = ''
        if cands:
            title = max(cands, key=lambda c: len(c['text']))['text']

        out.append(dict(index=a['idx'], title=title, page=page,
                        page_raw=page_cell, n_cells=len(region)))
    return out
