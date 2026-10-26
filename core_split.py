# -*- coding: utf-8 -*-
"""
PDF 智能拆分 — 核心流程（供 GUI / 命令行复用）
============================================
流程: 逐份 PDF -> 自动探测卷内目录页 -> 解析目录 -> 推算边界 -> 切分 -> 出报告

对外接口:
    run(in_dir, out_dir, with_cover=True, fast=True, log=print)
"""
import os
import io
import re
import gc
import json
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

# 打包成 exe 时, 模型随 exe 一起分发在 runtime/paddlex, 指向它
import sys as _sys
if getattr(_sys, 'frozen', False):
    _px = os.path.join(getattr(_sys, '_MEIPASS', ''), 'runtime', 'paddlex')
    if os.path.isdir(_px):
        os.environ.setdefault('PADDLE_PDX_CACHE_HOME', _px)

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, 'scripts')):
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

import numpy as np
from collections import Counter

TOC_KEYWORDS = ['卷内目录', '目录', '文件编号', '文件题名', '责任者']
# 页码格式: "7"、"194~196"、"117-128"、"117～128"
PAGE_RANGE_RE = re.compile(r'^\s*(\d{1,4})\s*[~～\-—至]\s*(\d{1,4})\s*$')


def safe_name(s, maxlen=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', '_', s or '')
    s = re.sub(r'\s+', '', s).strip(' .')
    return s[:maxlen] if s else 'untitled'


# ---------- 目录页自动探测 ----------

def detect_toc_pages(pdf_path, ocr, max_scan=8, log=print):
    """在前 max_scan 页里找出含「卷内目录」关键词的连续页范围(1-based, 闭区间)"""
    import pymupdf
    from PIL import Image
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    hit = []
    for i in range(min(max_scan, n)):
        pix = doc[i].get_pixmap(dpi=180)
        img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        res = ocr.predict(img)
        txt = ''
        for rr in res:
            txt += ''.join(rr.get('rec_texts', []))
        if any(k in txt for k in ('卷内目录', '文件题名', '文件编号')):
            hit.append(i + 1)
        del pix, img, res
        gc.collect()
    doc.close()
    if not hit:
        return None
    # 取连续段
    start = hit[0]
    end = hit[0]
    for p in hit[1:]:
        if p == end + 1:
            end = p
        else:
            break
    return (start, end)


# ---------- 页码识别 ----------

def read_page_number_ex(pdf_path, pi, ocr, dpi=120, cache=None):
    """识别物理页 pi(0-based) 的印刷页码, 返回 (页码或None, OCR原文)。

    dpi 默认 120: 实测页码是大字号、清晰, 120dpi 与 200dpi 识别结果完全一致,
    但整页渲染耗时省约 34%(见 docs 里的实测表)。
    """
    if cache is not None and pi in cache:
        return cache[pi]
    from page_map_engine import render_full, locate_page_number, crop_page_number
    arr = render_full(pdf_path, pi, dpi=dpi)
    loc = locate_page_number(arr)
    v, raw = None, ''
    if loc:
        im = crop_page_number(arr, loc['bbox'], scale=2)
        if im is not None:
            r2 = ocr.predict(np.array(im.convert('RGB')))
            raw = ''.join(''.join(rr.get('rec_texts', [])) for rr in r2).strip()
            v = int(raw) if re.fullmatch(r'\d{1,4}', raw) else None
    del arr
    gc.collect()
    if cache is not None:
        cache[pi] = (v, raw)
    return v, raw


def read_page_number(pdf_path, pi, ocr, dpi=120, cache=None):
    """只返回页码整数(内部走 read_page_number_ex)。"""
    v, _ = read_page_number_ex(pdf_path, pi, ocr, dpi=dpi, cache=cache)
    return v


def predict_local(obs, toc_page, k=6, main_off=None):
    """【快速版】局部最近邻 offset 加权众数(稳健处理分段 offset)。
    main_off: 全局主导 offset; 给了则先剔除偏离 >5 的离群观测点。"""
    if not obs:
        return None
    pool = obs
    if main_off is not None:
        clean = [o for o in obs if abs((o[0] - o[1]) - main_off) <= 5]
        if len(clean) >= 2:
            pool = clean
    pts = sorted(pool, key=lambda x: abs(x[1] - toc_page))
    w = Counter()
    for phy, pg in pts[:k]:
        w[phy - pg] += max(1, 10 - abs(pg - toc_page))
    if not w:
        return None
    if main_off is not None:
        cands = [o for o, _ in w.most_common()]
        near_off = min(cands, key=lambda o: abs(o - main_off))
        if abs(near_off - main_off) <= 5:
            return toc_page + near_off
    return toc_page + w.most_common(1)[0][0]


# ---------- 慢版(高精度)专用: 分段拟合 + 局部预测 ----------

def local_mode(seq, k=5):
    """滑动窗口众数平滑(抗 OCR 噪声)"""
    n = len(seq)
    out = []
    for i in range(n):
        lo = max(0, i - k // 2)
        hi = min(n, i + k // 2 + 1)
        vals = [seq[j] for j in range(lo, hi)]
        out.append(Counter(vals).most_common(1)[0][0])
    return out


def fit_segments(obs, min_seg=4, min_page_span=3):
    """把观测点拟合成若干"分段 offset"区间。
    与 scripts/offset_fitter.py 的 fit_segments 同算法。"""
    if not obs:
        return []
    obs = sorted(obs)
    offs = [p - v for p, v in obs]
    smooth = local_mode(offs, k=5)
    raw_segs = []
    s_i = 0
    for i in range(1, len(smooth)):
        if smooth[i] != smooth[s_i]:
            raw_segs.append((s_i, i - 1, smooth[s_i]))
            s_i = i
    raw_segs.append((s_i, len(smooth) - 1, smooth[s_i]))
    cand = []
    for a, b, off in raw_segs:
        cand.append(dict(start_phy=obs[a][0], end_phy=obs[b][0], offset=off,
                         page_lo=obs[a][1], page_hi=obs[b][1], n_pts=b - a + 1))
    kept = [s for s in cand
            if s['n_pts'] >= min_seg and (s['page_hi'] - s['page_lo']) >= min_page_span]
    if not kept:
        all_off = Counter(offs).most_common(1)[0][0]
        return [dict(start_phy=obs[0][0], end_phy=obs[-1][0], offset=all_off,
                     page_lo=obs[0][1], page_hi=obs[-1][1], n_pts=len(obs))]
    merged = [kept[0]]
    for s in kept[1:]:
        if abs(s['offset'] - merged[-1]['offset']) <= 2:
            merged[-1]['end_phy'] = s['end_phy']
            merged[-1]['page_hi'] = s['page_hi']
            merged[-1]['n_pts'] += s['n_pts']
        else:
            merged.append(s)
    return merged


def predict_local_seg(obs, toc_page, k=8):
    """【慢版】局部最近邻 offset 预测。
    返回 (预测物理页, 局部offset, 参考点) 或 (None, None, None)。
    与 scripts/offset_fitter.py 的 predict_local 同算法。"""
    if not obs:
        return None, None, None
    pts = sorted(obs, key=lambda x: abs(x[1] - toc_page))
    near = pts[:k]
    weights = Counter()
    for phy, pg in near:
        d = abs(pg - toc_page)
        weights[phy - pg] += max(1, 10 - d)
    off = weights.most_common(1)[0][0]
    return toc_page + off, off, near[:3]


def adaptive_sample_pis(N, n_coarse=12, probe=6, gap_thresh=12):
    """【慢版提速】自适应抽样页(0-based)。

    旧做法: 固定 step=N//60 密抽样(125页 -> 63 点), 绝大多数是冗余的。
    新做法: 第一轮稀疏 n_coarse 点定主 offset; 第二轮仅在"可疑处"补点——
      可疑 = 相邻观测 offset 不一致, 或两点页码跨度 > gap_thresh。
    返回 [(pi, round)] 列表, round 0=粗抽, 1=加密(便于日志区分)。

    这不是降低精度: offset 分段点本就需要在跳变附近加密, 稀疏点找不到跳变时
    靠第二轮补齐; 逐条锚点验证(慢版核心)完全不变。
    """
    if N <= 0:
        return []
    n_coarse = min(n_coarse, N)
    step = max(1, N // n_coarse)
    coarse = sorted(set(range(0, N, step)) | {N - 1})
    out = [(pi, 0) for pi in coarse]

    # 第二轮的候选区间: 相邻粗抽点之间
    extra = []
    for a, b in zip(coarse, coarse[1:]):
        span = b - a
        # 区间太短 -> 不必补; 太长 -> 补 probe 个中间点
        if span <= max(2, step):
            continue
        m = min(probe, span - 1)
        if m <= 0:
            continue
        for j in range(1, m + 1):
            extra.append(a + round(span * j / (m + 1)))
    for pi in sorted(set(extra)):
        if pi not in coarse and 0 <= pi < N:
            out.append((pi, 1))
    return sorted(out)


def refine_sample_pis(obs, sample_pis, N, probe=4, gap_thresh=12):
    """根据第一轮观测结果, 判断是否需要在某区间加密(带 OCR 后的二次加密)。

    obs: [(物理页1based, 页码)] 已观测
    sample_pis: 当前已抽的 (pi0based, round)
    返回: 需要补抽的 pi 列表(0based)。空列表=无需加密。
    """
    if len(obs) < 3:
        return []
    offs = [round(q - p) for q, p in obs]
    main_off = Counter(offs).most_common(1)[0][0]
    have = {pi for pi, _ in sample_pis}
    need = []
    obs_sorted = sorted(obs)
    for (pa, va), (pb, vb) in zip(obs_sorted, obs_sorted[1:]):
        oa, ob = pa - va, pb - vb
        # 相邻点 offset 不一致 -> 跳变区, 加密
        suspicious = (abs(oa - main_off) > 1 or abs(ob - main_off) > 1)
        # 跨度虽大但 offset 一致 -> 不必加密(节省 OCR)
        if not suspicious:
            continue
        span = pb - pa
        if span <= 2:
            continue
        m = min(probe, span - 1)
        for j in range(1, m + 1):
            cand = pa + round(span * j / (m + 1)) - 1   # ->0based
            if cand not in have and 0 <= cand < N:
                need.append(cand)
    return sorted(set(need))


# ---------- 单份拆分 ----------

def split_one(pdf_path, out_dir, with_cover=True, fast=True, log=print):
    import pymupdf
    from paddleocr import PaddleOCR
    from toc_parser import ocr_items, cluster_rows, extract_toc_rows

    name = os.path.splitext(os.path.basename(pdf_path))[0]
    log(f'\n===== {name} =====')

    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    doc = pymupdf.open(pdf_path)
    N = doc.page_count
    doc.close()
    log(f'总页数 {N}')

    # 1) 目录页
    toc_range = detect_toc_pages(pdf_path, ocr, log=log)
    if not toc_range:
        log('!! 未找到「卷内目录」页, 跳过')
        return None
    log(f'目录页: 物理 {toc_range[0]}~{toc_range[1]}')

    # 2) 解析目录
    from PIL import Image
    doc = pymupdf.open(pdf_path)
    rows_all = []
    for i in range(toc_range[0] - 1, toc_range[1]):
        pix = doc[i].get_pixmap(dpi=250)
        img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        res = ocr.predict(img)
        items = ocr_items(res)
        rows = cluster_rows(items, page_h=img.shape[0])
        for r in extract_toc_rows(rows, img.shape[1]):
            r['src_page'] = i + 1
            rows_all.append(r)
        del pix, img, res, items, rows
        gc.collect()
    doc.close()
    entries = sorted([e for e in rows_all if e.get('page')], key=lambda e: e['index'])
    for e in entries:
        e['start_page'] = e['page'][0]
        e['end_page'] = e['page'][1]
    log(f'目录解析: {len(entries)} 条')
    if not entries:
        log('!! 目录为空, 跳过')
        return None

    # 3) 建页码映射
    #    快速版: 固定抽样 12 页(不做逐条验证)
    #    高精度版: 自适应两轮抽样 —— 稀疏12点定主offset, 仅在offset跳变区加密
    #              (旧版是固定 N//60 密抽样, 冗余严重; 精度不降, 见 docs)
    pn_cache = {}
    obs = []
    if fast:
        n_samples = 12
        step = max(1, N // (n_samples + 1))
        sample_pis = list(range(0, N, step))
        for pi in sample_pis:
            v = read_page_number(pdf_path, pi, ocr, cache=pn_cache)
            if v is not None:
                obs.append((pi + 1, v))
    else:
        plan = adaptive_sample_pis(N)
        n_coarse = sum(1 for _, r in plan if r == 0)
        log(f'  自适应抽样: 第1轮 {n_coarse} 点'
            f'(旧版密抽样 {len(range(0, N, max(1, N // 60)))} 点)')
        for pi, _r in plan:
            v = read_page_number(pdf_path, pi, ocr, cache=pn_cache)
            if v is not None:
                obs.append((pi + 1, v))
        # 第二轮: 若发现 offset 跳变区, 就地加密(最多再补几十点)
        need = refine_sample_pis(obs, plan, N)
        if need:
            log(f'  发现 offset 跳变迹象 -> 加密补抽 {len(need)} 点')
            for pi in need:
                v = read_page_number(pdf_path, pi, ocr, cache=pn_cache)
                if v is not None:
                    obs.append((pi + 1, v))
        obs = sorted(set(obs))
    if not obs:
        log('!! 无有效页码观测, 跳过')
        return None
    segs = fit_segments(obs) if not fast else []
    anomalies = []
    log(f'页码抽样: {len(obs)} 个有效点, offset 众数 '
        f'{Counter(round(q - p) for q, p in obs).most_common(2)}')
    if segs:
        log(f'  分段拟合 {len(segs)} 段: ' + ' | '.join(
            f"物理{s['start_phy']}~{s['end_phy']} offset={s['offset']}"
            for s in segs))

    if fast:
        # ---------- 快速版: 直接推算边界(无逐条 OCR 验证) ----------
        main_off = Counter(round(q - p) for q, p in obs).most_common(1)[0][0]
        clean_obs = [o for o in obs if abs((o[0] - o[1]) - main_off) <= 5]
        if len(clean_obs) < 3:
            clean_obs = obs
        raw = []
        for e in entries:
            cand = predict_local(obs, e['start_page'], main_off=main_off)
            cand = int(cand) if cand else 1
            cand = max(1, min(cand, N))
            dev = abs((cand - e['start_page']) - main_off) > 1
            nearest_d = min((abs(o[1] - e['start_page']) for o in clean_obs),
                            default=9999)
            sparse = nearest_d > 40
            raw.append(dict(index=e['index'], title=e['title'],
                            start_page=e['start_page'], end_page=e['end_page'],
                            cand_phy=cand, suspect=bool(dev or sparse)))
    else:
        # ---------- 高精度版: 锚点预测 + 逐条 OCR 验证(带窗口搜索) ----------
        SEARCH_WIN = 3
        raw = []

        def read_cached(pi):        # pi: 1-based 物理页
            if not (1 <= pi <= N):
                return None, ''
            return read_page_number_ex(pdf_path, pi - 1, ocr, cache=pn_cache)

        for e in entries:
            sp, ep = e['start_page'], e['end_page']
            pred, loc_off, refs = predict_local_seg(obs, sp)
            if pred is None:
                raw.append(dict(index=e['index'], title=e['title'],
                                start_page=sp, end_page=ep, cand_phy=1,
                                pred_phy=None, confirm=False, suspect=True))
                continue
            found, found_raw = pred, ''
            v0, raw0 = read_cached(pred)
            found_raw = raw0
            if v0 != sp:
                for k in range(1, SEARCH_WIN + 1):
                    hit = False
                    for d in (-k, k):
                        cand = pred + d
                        if not (1 <= cand <= N):
                            continue
                        v, rw = read_cached(cand)
                        if v == sp:
                            found, found_raw, hit = cand, rw, True
                            break
                    if hit:
                        break
            confirm = (found_raw == str(sp))
            mark = 'OK' if confirm else ('SHIFT' if found != pred else '?')
            log(f"  #{e['index']:>4} 页{sp:>4}->预测{pred:>4} 实锚{found:>4} "
                f"got={found_raw!r} {mark}")
            if found != pred:
                anomalies.append(dict(type='ANCHOR_SHIFT', index=e['index'],
                                      start=sp, pred=pred, found=found))
            if not confirm:
                anomalies.append(dict(type='UNCONFIRMED', index=e['index'],
                                      start=sp, phy=found, got=found_raw))
            raw.append(dict(index=e['index'], title=e['title'],
                            start_page=sp, end_page=ep, cand_phy=found,
                            pred_phy=pred, got_raw=found_raw,
                            confirm=confirm, suspect=False))

    # 4a) 重复登记(只向前看)
    for i, b in enumerate(raw):
        for j in range(i + 1, len(raw)):
            if raw[j]['cand_phy'] == b['cand_phy']:
                b['skip'] = True
                b['dup_of'] = raw[j]['index']
                anomalies.append(dict(type='TOC_DUPLICATE_ROW', index=b['index'],
                                      start=b['start_page'], dup_of=b['dup_of']))
                log(f"  x#{b['index']} 重复登记(与 #{b['dup_of']}) -> 跳过")
                break

    # 4b) 页码倒退/非单调 -> 兜底
    eff_prev = None
    for i, b in enumerate(raw):
        if b.get('skip'):
            continue
        if eff_prev is None:
            eff_prev = b['cand_phy']; continue
        prev = next((raw[k] for k in range(i - 1, -1, -1)
                     if not raw[k].get('skip')), None)
        if prev and (b['start_page'] < prev['end_page'] or b['cand_phy'] <= eff_prev):
            b['fixed'] = True
            b['cand_phy'] = eff_prev + 1
            b['suspect'] = True
            anomalies.append(dict(type='TOC_BACKWARD_FIXED', index=b['index'],
                                  start=b['start_page'], prev_end=prev['end_page']))
            log(f"  !#{b['index']} 页码异常({b['start_page']}) -> 连续性推定")
        eff_prev = b['cand_phy']

    # 4c) 异常段区间均分
    i = 0
    while i < len(raw):
        if raw[i].get('skip') or not raw[i].get('fixed'):
            i += 1; continue
        j = i
        while (j + 1 < len(raw) and raw[j + 1].get('fixed')
               and not raw[j + 1].get('skip')):
            j += 1
        P = next((raw[k]['cand_phy'] for k in range(i - 1, -1, -1)
                  if not raw[k].get('fixed') and not raw[k].get('skip')), 0)
        Q = next((raw[k]['cand_phy'] for k in range(j + 1, len(raw))
                  if not raw[k].get('fixed') and not raw[k].get('skip')), N + 1)
        cnt = j - i + 1
        per = max(1, max(cnt, Q - P - 1) // cnt)
        for k in range(cnt):
            raw[i + k]['cand_phy'] = P + 1 + k * per
            raw[i + k]['suspect'] = True
        log(f"  >> 异常段 #{raw[i]['index']}~#{raw[j]['index']} -> 每条{per}页")
        i = j + 1

    # 4d) 单调清理
    pe = None
    for b in raw:
        if b.get('skip'):
            continue
        if pe is not None and b['cand_phy'] <= pe:
            b['cand_phy'] = pe + 1
            b['suspect'] = True
        pe = b['cand_phy']

    bounds = [b for b in raw if not b.get('skip')]
    for b in bounds:
        b['start_phy'] = b['cand_phy']
    for i, b in enumerate(bounds):
        b['end_phy'] = bounds[i + 1]['start_phy'] - 1 if i + 1 < len(bounds) else None

    # 5) 拆分
    sub = os.path.join(out_dir, name)
    os.makedirs(sub, exist_ok=True)
    lastb = bounds[-1]
    if lastb.get('end_page') and lastb['end_page'] > lastb['start_page']:
        body_last = min(lastb['start_phy'] + (lastb['end_page'] - lastb['start_page']), N)
    else:
        body_last = N - 1 if N > 1 else N

    doc = pymupdf.open(pdf_path)
    produced = []
    for bi, b in enumerate(bounds):
        s = b['start_phy']
        e = min(b['end_phy'] if b['end_phy'] else body_last, body_last)
        if bi == 0 and with_cover and s > 1:
            s = 1
        if e < s:
            continue
        # 命名: 纯产出序号 001/002/... (不含标题、不含任何后缀)
        fn = f"{len(produced) + 1:03d}.pdf"
        nd = pymupdf.open()
        nd.insert_pdf(doc, from_page=s - 1, to_page=e - 1)
        nd.save(os.path.join(sub, fn), garbage=3, deflate=True)
        nd.close()
        produced.append(dict(file=fn, index=b['index'], start_phy=s, end_phy=e,
                             n_pages=e - s + 1, suspect=b.get('suspect', False)))
    doc.close()
    log(f'产出 {len(produced)} 个 PDF '
        f'(待核 {sum(1 for p in produced if p["suspect"])} 个) -> {sub}')
    return dict(name=name, total=N, toc_range=toc_range, entries=len(entries),
                produced=produced, bounds=bounds, mode='fast' if fast else 'accurate',
                anomalies=anomalies, segments=segs,
                confirmed=sum(1 for b in bounds if b.get('confirm')))


# ---------- 批量入口 ----------

def run(in_dir, out_dir, with_cover=True, fast=True, log=print):
    t0 = time.time()
    pdfs = sorted([f for f in os.listdir(in_dir) if f.lower().endswith('.pdf')])
    if not pdfs:
        log('!! 输入文件夹里没有 PDF')
        return
    log(f'发现 {len(pdfs)} 个 PDF, 开始处理…')
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for f in pdfs:
        try:
            r = split_one(os.path.join(in_dir, f), out_dir,
                          with_cover=with_cover, fast=fast, log=log)
            if r:
                results.append(r)
        except Exception as e:
            log(f'!! {f} 处理失败: {e!r}')
            log(traceback.format_exc())
    # 报告
    lines = ['# PDF 智能拆分 — 结果报告', '',
             f'输入: {in_dir}', f'输出: {out_dir}',
             f"模式: {'快速（抽样12页）' if fast else '高精度（逐条OCR验证）'}",
             f'耗时: {time.time()-t0:.0f} 秒', '']
    lines.append('| PDF | 总页 | 目录条目 | 产出PDF | 待核 | 说明 |')
    lines.append('|---|---|---|---|---|---|')
    for r in results:
        sus = sum(1 for p in r['produced'] if p['suspect'])
        note = ''
        if r.get('mode') == 'accurate':
            note = f"OCR确认 {r.get('confirmed', 0)}/{len(r['produced'])}"
        else:
            note = '抽样推算'
        lines.append(f"| {r['name']} | {r['total']} | {r['entries']} | "
                     f"{len(r['produced'])} | {sus} | {note} |")
    # 待核清单
    sus_items = []
    for r in results:
        for p in r['produced']:
            if p['suspect']:
                sus_items.append((r['name'], p['file']))
    if sus_items:
        lines += ['', '## 待核清单（文件名纯编号，此处按序号列出）', '']
        for nm, fn in sus_items:
            lines.append(f'- `{nm}/{fn}`')
    open(os.path.join(out_dir, '拆分报告.md'), 'w', encoding='utf-8').write(
        '\n'.join(lines))
    log(f'\n合计 {sum(len(r["produced"]) for r in results)} 个 PDF, '
        f'耗时 {time.time()-t0:.0f} 秒')
    log(f'报告: {os.path.join(out_dir, "拆分报告.md")}')
    return results
