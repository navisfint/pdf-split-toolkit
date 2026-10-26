"""
对齐校验与拆分引擎
==================
输入: 目录条目 (含起始页码) + 正文页码映射(物理页->档案页码)
输出: 按目录条目切分的多个 PDF

决策依据（用户已拍板）:
  1. 以【正文页码】为准: 目录给的"起始页码"用于找到正文里那一页;
     最终切分边界物理页 = "档案页码 == 目录起始页码" 的物理页
  2. 目录页码倒退 -> 遇错即停: 该行及之后的行全部不输出，标记异常
  3. 正文末尾超出目录末项 -> 差异部分单独成档 (命名 999_未列入目录.pdf)

切分规则:
  条目 i 的物理页范围 = [ start_phy(i), start_phy(i+1) - 1 ]
  最后一条 = [ start_phy(n), 正文最后一页 ]
"""
import os, json, re


def _safe_name(s, maxlen=60):
    """题名清洗为安全文件名"""
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', '_', s or '')
    s = re.sub(r'\s+', ' ', s).strip(' .')
    return s[:maxlen] if s else 'untitled'


def load_page_map(page_map_json):
    """读页码映射 -> {物理页:int -> 档案页码:int}"""
    d = json.load(open(page_map_json, encoding='utf-8'))
    return {int(k): int(v) for k, v in d['page_map'].items()}


def find_phy_by_page(page_map, target_page, tol=2):
    """
    在映射里找"档案页码 == target_page"的物理页。
    页码可能因 OCR 误差有偏差，允许 ±tol 匹配；优先精确。
    返回 (phy, exact) 或 (None, False)
    """
    exact = [p for p, v in page_map.items() if v == target_page]
    if exact:
        return min(exact), True
    # 容差匹配，取最接近的
    best, bestd = None, 1e9
    for p, v in page_map.items():
        d = abs(v - target_page)
        if d <= tol and d < bestd:
            best, bestd = p, d
    return (best, False) if best else (None, False)


def align(entries, page_map, body_start_phy=1):
    """
    把目录条目对齐到物理页。
    entries: [{index, title, page(start,end), ...}]
    page_map: {phy: 档案页码}
    返回 dict(boundaries=[{index,title,start_phy,end_phy,start_page,exact,note}],
              anomalies=[...], stop_at=None)
    """
    anomalies = []
    ordered = sorted(entries, key=lambda e: e['index'])

    # --- 1) 目录页码倒退检测 -> 遇错即停 ---
    stop_at = None
    last_end = None
    valid = []
    for e in ordered:
        pg = e.get('page')
        if not pg:
            anomalies.append(dict(type='NO_PAGE', index=e['index'], title=e['title']))
            continue
        start = pg[0]
        if last_end is not None and start < last_end:
            anomalies.append(dict(
                type='TOC_BACKWARD', index=e['index'], title=e['title'],
                start=start, prev_end=last_end,
                detail=f'目录页码倒退: 第{e["index"]}项起始页{start} < 前项结束页{last_end}'))
            stop_at = e['index']
            break
        if last_end is not None and start == last_end:
            anomalies.append(dict(type='TOC_OVERLAP', index=e['index'], title=e['title'],
                                  start=start, detail='与前项结束页重合'))
        valid.append(e)
        last_end = pg[1]

    # --- 2) 对齐到物理页 ---
    boundaries = []
    for e in valid:
        start_page = e['page'][0]
        phy, exact = find_phy_by_page(page_map, start_page)
        if phy is None:
            anomalies.append(dict(type='ANCHOR_NOT_FOUND', index=e['index'],
                                  title=e['title'], start_page=start_page,
                                  detail=f'正文中找不到页码 {start_page}'))
            continue
        if not exact:
            anomalies.append(dict(type='ANCHOR_INEXACT', index=e['index'],
                                  title=e['title'], start_page=start_page,
                                  matched_page=page_map[phy],
                                  detail=f'页码 {start_page} 近似匹配到 {page_map[phy]}'))
        boundaries.append(dict(index=e['index'], title=e['title'],
                               start_phy=phy, start_page=start_page, exact=exact))

    # --- 3) 单调性修正: 物理页必须递增 ---
    fixed = []
    for b in boundaries:
        if fixed and b['start_phy'] <= fixed[-1]['start_phy']:
            anomalies.append(dict(type='BOUNDARY_NONMONO', index=b['index'],
                                  title=b['title'], start_phy=b['start_phy'],
                                  prev_phy=fixed[-1]['start_phy'],
                                  detail='物理页边界非递增，已跳过该边界'))
            continue
        fixed.append(b)
    boundaries = fixed

    # --- 4) 计算每条的结束物理页 ---
    for i, b in enumerate(boundaries):
        if i + 1 < len(boundaries):
            b['end_phy'] = boundaries[i + 1]['start_phy'] - 1
        else:
            b['end_phy'] = None   # 最后一条: 到正文末(调用方填)
    return dict(boundaries=boundaries, anomalies=anomalies, stop_at=stop_at)


def split_pdf(src, boundaries, body_last_phy, out_dir,
              front_pages=1, name_fmt='{index:03d}_{title}.pdf'):
    """
    执行拆分。
    body_last_phy: 正文最后一页的物理页号(1-based)
    front_pages: 正文之前的封面/目录页数(不参与拆分)
    返回 产出文件列表 + 异常
    """
    import pymupdf
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(src)
    produced, errs = [], []

    bounds = [b for b in boundaries if b['start_phy'] <= body_last_phy]
    for i, b in enumerate(bounds):
        s = b['start_phy']
        e = b['end_phy'] if b.get('end_phy') else body_last_phy
        e = min(e, body_last_phy)
        if e < s:
            errs.append(dict(type='EMPTY_RANGE', index=b['index'], start=s, end=e))
            continue
        name = name_fmt.format(index=b['index'], title=_safe_name(b['title']))
        dst = os.path.join(out_dir, name)
        nd = pymupdf.open()
        nd.insert_pdf(doc, from_page=s - 1, to_page=e - 1)
        nd.save(dst, garbage=3, deflate=True)
        nd.close()
        produced.append(dict(path=dst, index=b['index'], title=b['title'],
                             start_phy=s, end_phy=e, n_pages=e - s + 1))

    # 末尾差异单独成档
    if bounds:
        last_end = bounds[-1]['end_phy'] or body_last_phy
        if last_end < body_last_phy:
            dst = os.path.join(out_dir, '999_未列入目录.pdf')
            nd = pymupdf.open()
            nd.insert_pdf(doc, from_page=last_end, to_page=body_last_phy - 1)
            nd.save(dst, garbage=3, deflate=True)
            nd.close()
            produced.append(dict(path=dst, index=999, title='未列入目录',
                                 start_phy=last_end + 1, end_phy=body_last_phy,
                                 n_pages=body_last_phy - last_end))
    doc.close()
    return produced, errs
