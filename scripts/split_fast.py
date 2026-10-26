# -*- coding: utf-8 -*-
"""
PDF 智能拆分 — 快速版 (v-fast)
==============================
与 split_main.py (高精度版) 的区别:
  高精度版: 抽样求分段 offset -> 每条目录行都做 OCR 锚点验证 -> 慢但准
  快速版  : 少量抽样建页码映射 -> 线性插值直接推算所有边界 -> 快 5~10 倍
            (放弃逐条 OCR 验证, 只做少量校验; 有问题的行照样标记 _待核)

速度来源:
  1. 抽样页数从 N/100 降到 12 页固定
  2. 不做逐条锚点验证 (最大开销, 53 条 = 53 次 OCR 往返)
  3. 用 (物理页, 页码) 观测点拟合"物理页 = 页码 + offset"的分段线性关系

用法:
  python split_fast.py <tag> [--dry-run] [--samples N] [--dpi N]
"""
import os, sys, json, gc, re
import numpy as np
from collections import Counter

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
OUT_ROOT = r'F:\workBuddy\PDF智能拆分\output'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')


def safe_name(s, maxlen=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', '_', s or '')
    s = re.sub(r'\s+', '', s).strip(' .')
    return s[:maxlen] if s else 'untitled'


def build_map(obs):
    """obs: [(物理页, 页码)] -> 返回 [(页码, 物理页)] 升序观测点(去重取中位)"""
    pts = sorted([(p, ph) for ph, p in obs])       # 按页码排序: (页码, 物理页)
    dedup = {}
    for p, ph in pts:
        dedup.setdefault(p, []).append(ph)
    xs = sorted(dedup)
    ys = [float(np.median(dedup[p])) for p in xs]
    return xs, ys


def predict_local(obs, toc_page, k=6, main_off=None):
    """局部最近邻 offset 加权众数 —— 与高精度版同算法(稳健处理分段 offset)。
    取页码最接近的 k 个观测点, 对 (物理页-页码) 做权重投票, 选众数。
    main_off: 全局主导 offset; 若给了, 则先剔除偏离 >5 的离群观测点再投票。"""
    if not obs:
        return None
    pool = obs
    if main_off is not None:
        clean = [o for o in obs if abs((o[0] - o[1]) - main_off) <= 5]
        if len(clean) >= 2:
            pool = clean
    pts = sorted(pool, key=lambda x: abs(x[1] - toc_page))
    near = pts[:k]
    w = Counter()
    for phy, pg in near:
        d = abs(pg - toc_page)
        w[phy - pg] += max(1, 10 - d)
    if not w:
        return None
    # 若给了 main_off 且众数与它差异过大, 用众数里最接近 main_off 的
    if main_off is not None:
        cands = [o for o, _ in w.most_common()]
        near_off = min(cands, key=lambda o: abs(o - main_off))
        if abs(near_off - main_off) <= 5:
            return toc_page + near_off
    return toc_page + w.most_common(1)[0][0]


def interp_phy(xs, ys, page):
    """(保留) 页码线性插值, 仅用于 predict_local 失败时的兜底"""
    if not xs:
        return None
    import bisect
    if page <= xs[0]:
        k = (ys[1] - ys[0]) / (xs[1] - xs[0]) if len(xs) >= 2 and xs[1] != xs[0] else 1.0
        return ys[0] + k * (page - xs[0])
    if page >= xs[-1]:
        k = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) >= 2 and xs[-1] != xs[-2] else 1.0
        return ys[-1] + k * (page - xs[-1])
    i = bisect.bisect_right(xs, page) - 1
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (page - x0) / (x1 - x0)


def main():
    tag = sys.argv[1]
    dry = '--dry-run' in sys.argv
    n_samples = 12
    if '--samples' in sys.argv:
        n_samples = int(sys.argv[sys.argv.index('--samples') + 1])
    dpi = 200
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi') + 1])

    import pymupdf
    from paddleocr import PaddleOCR
    from page_map_engine import render_full, locate_page_number, crop_page_number

    toc = json.load(open(os.path.join(PROBE, f'toc_{tag}.json'), encoding='utf-8'))
    if isinstance(toc, dict):
        toc = toc[tag]
    entries = sorted([e for e in toc if e.get('page')], key=lambda e: e['index'])
    for e in entries:
        e['start_page'] = e['page'][0]
        e['end_page'] = e['page'][1]

    src = os.path.join(D, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}.pdf')
    doc = pymupdf.open(src); N = doc.page_count; doc.close()
    print(f'[{tag}] 总页={N}, 目录条目={len(entries)}, 快速版抽样={n_samples}',
          flush=True)

    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    def read_pn(pi):          # pi: 0-based
        arr = render_full(src, pi, dpi=dpi)
        loc = locate_page_number(arr)
        v = None
        if loc:
            im = crop_page_number(arr, loc['bbox'], scale=2)
            if im is not None:
                r2 = ocr.predict(np.array(im.convert('RGB')))
                raw = ''
                for rr in r2:
                    raw += ''.join(rr.get('rec_texts', []))
                raw = raw.strip()
                v = int(raw) if re.fullmatch(r'\d{1,4}', raw) else None
        del arr; gc.collect()
        return v

    # ---- 1) 均匀抽样建映射 ----
    obs = []
    step = max(1, N // (n_samples + 1))
    for pi in range(0, N, step):
        v = read_pn(pi)
        if v is not None:
            obs.append((pi + 1, v))
    if not obs:
        print('!! 无有效页码观测, 中止'); return
    xs, ys = build_map(obs)
    offs = [round(ph - p) for p, ph in obs]
    oc = Counter(offs).most_common(3)
    print(f'[1] 抽样 {len(obs)}/{N//step+1} 命中, offset 前3: {oc}', flush=True)

    # ---- 2) 局部最近邻 offset 推算每条起始物理页 ----
    # 轻量校验: 若该行推断出的 offset 与"最近邻 offset"差异 > 1, 标记待核
    main_off = Counter(round(ph - p) for ph, p in obs).most_common(1)[0][0]
    raw_bounds, anomalies = [], []
    for e in entries:
        sp, ep = e['start_page'], e['end_page']
        cand = predict_local(obs, sp, k=6, main_off=main_off)
        if cand is None:                       # 兜底: 线性插值
            cand = interp_phy(xs, ys, sp)
        if cand is None:
            anomalies.append(dict(type='NO_SEGMENT', index=e['index'],
                                  start=sp, title=e['title']))
            raw_bounds.append(dict(index=e['index'], title=e['title'],
                                   start_page=sp, end_page=ep, cand_phy=1,
                                   suspect=True, confirm=False))
            continue
        cand = int(round(cand))
        cand = max(1, min(cand, N))
        # 嫌疑判定(快版特有, 放弃纠错但保留提示):
        #  a) 该行 offset 偏离主导 offset > 1
        #  b) 该行附近 k 个观测点里 offset 不唯一(说明正处于分段 offset 的过渡区,
        #     此时抽样法容易差 1~2 页)
        dev = abs((cand - sp) - main_off) > 1
        # 稳定性: 只在"已剔除离群点"的观测集上判断。
        # 若最近的观测点距本行页码很远(>40 页), 说明该区域没抽到点,
        # 偏移全靠外推 -> 不可靠, 标待核。
        clean = [o for o in obs if abs((o[0] - o[1]) - main_off) <= 5]
        if len(clean) < 3:
            clean = obs
        nearest_d = min((abs(o[1] - sp) for o in clean), default=9999)
        sparse = nearest_d > 40
        susp = bool(dev or sparse)
        if susp:
            reason = []
            if dev:
                reason.append('offset偏离')
            if sparse:
                reason.append(f'抽样稀疏({nearest_d}页外)')
            anomalies.append(dict(type='FAST_OFFSET_DEVIATION', index=e['index'],
                                  start=sp, phy=cand, title=e['title'],
                                  reason='/'.join(reason)))
        raw_bounds.append(dict(index=e['index'], title=e['title'],
                               start_page=sp, end_page=ep, cand_phy=cand,
                               pred_phy=cand, suspect=susp, confirm=not susp,
                               fast=True))

    # ---- 3) 异常处理(与高精度版同规则) ----
    n_all = len(raw_bounds)
    # 3a. 重复登记: 只向前看, 后者优先保留
    for i, b in enumerate(raw_bounds):
        if b.get('skip'):
            continue
        for j in range(i + 1, n_all):
            o = raw_bounds[j]
            if o.get('skip'):
                continue
            if o['cand_phy'] == b['cand_phy']:
                b['skip'] = True
                b['dup_of'] = o['index']
                break

    for b in raw_bounds:
        if b.get('skip'):
            anomalies.append(dict(type='TOC_DUPLICATE_ROW', index=b['index'],
                                  start=b['start_page'], dup_of=b['dup_of'],
                                  title=b['title']))
            print(f"  x#{b['index']:>4} 重复登记(与 #{b['dup_of']} 同页 "
                  f"物理{b['cand_phy']}) -> 跳过", flush=True)

    # 3b. 页码倒退 / 物理页非单调 -> 兜底
    eff_prev = None
    for i, b in enumerate(raw_bounds):
        if b.get('skip'):
            continue
        if eff_prev is None:
            b['fixed'] = False
            eff_prev = b['cand_phy']
            continue
        prev = None
        for k in range(i - 1, -1, -1):
            if not raw_bounds[k].get('skip'):
                prev = raw_bounds[k]; break
        backward = b['start_page'] < prev['end_page']
        nonmono = b['cand_phy'] <= eff_prev
        b['fixed'] = bool(backward or nonmono)
        if b['fixed']:
            b['cand_phy'] = eff_prev + 1
            anomalies.append(dict(type='TOC_BACKWARD_FIXED', index=b['index'],
                                  start=b['start_page'], prev_end=prev['end_page'],
                                  title=b['title'], suspect=True))
            print(f"  !#{b['index']:>4} 页码异常({b['start_page']}) -> 连续性推定",
                  flush=True)
        eff_prev = b['cand_phy']

    # 3c. 异常段区间均分
    i = 0
    while i < n_all:
        if raw_bounds[i].get('skip') or not raw_bounds[i].get('fixed'):
            i += 1; continue
        j = i
        while (j + 1 < n_all and raw_bounds[j + 1].get('fixed')
               and not raw_bounds[j + 1].get('skip')):
            j += 1
        P = 0
        for k in range(i - 1, -1, -1):
            if not raw_bounds[k].get('fixed') and not raw_bounds[k].get('skip'):
                P = raw_bounds[k]['cand_phy']; break
        Q = N + 1
        for k in range(j + 1, n_all):
            if not raw_bounds[k].get('fixed') and not raw_bounds[k].get('skip'):
                Q = raw_bounds[k]['cand_phy']; break
        cnt = j - i + 1
        span = max(cnt, Q - P - 1)
        per = max(1, span // cnt)
        for k in range(cnt):
            b = raw_bounds[i + k]
            b['cand_phy'] = P + 1 + k * per
            b['suspect'] = True
        print(f"  >> 异常段 #{raw_bounds[i]['index']}~#{raw_bounds[j]['index']}: "
              f"区间({P},{Q}) 共{cnt}条 -> 每条{per}页", flush=True)
        i = j + 1

    # 3d. 单调性清理
    prev_eff = None
    for i in range(n_all):
        if raw_bounds[i].get('skip'):
            continue
        if prev_eff is not None and raw_bounds[i]['cand_phy'] <= prev_eff:
            raw_bounds[i]['cand_phy'] = prev_eff + 1
            raw_bounds[i]['suspect'] = True
        prev_eff = raw_bounds[i]['cand_phy']

    bounds = [b for b in raw_bounds if not b.get('skip')]
    for b in bounds:
        b['start_phy'] = b['cand_phy']
    for i, b in enumerate(bounds):
        b['end_phy'] = (bounds[i + 1]['start_phy'] - 1
                        if i + 1 < len(bounds) else None)
    sus = sum(1 for b in bounds if b['suspect'])
    dup = sum(1 for b in raw_bounds if b.get('skip'))
    print(f'\n[2] 边界={len(bounds)}(跳过{dup}), 存疑={sus}, 异常={len(anomalies)}',
          flush=True)

    if dry:
        json.dump(dict(tag=tag, obs=obs, bounds=bounds, anomalies=anomalies,
                       total_pages=N, mode='fast'),
                  open(os.path.join(PROBE, f'split_fast_plan_{tag}.json'), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
        print('DRY_RUN_DONE', flush=True)
        return

    # ---- 4) 拆分 ----
    out_dir = os.path.join(OUT_ROOT, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}')
    if '--out' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--out') + 1]
    os.makedirs(out_dir, exist_ok=True)
    lastb = bounds[-1]
    if lastb.get('end_page') and lastb['end_page'] > lastb['start_page']:
        body_last = min(lastb['start_phy'] + (lastb['end_page'] - lastb['start_page']), N)
    else:
        body_last = N - 1 if N > 1 else N
    doc = pymupdf.open(src)
    produced = []
    for bi, b in enumerate(bounds):
        s = b['start_phy']
        e = min(b['end_phy'] if b['end_phy'] else body_last, body_last)
        if bi == 0 and s > 1:       # 封面 + 目录并入首个 PDF
            s = 1
            b['with_front'] = True
        if e < s:
            anomalies.append(dict(type='EMPTY_RANGE', index=b['index'])); continue
        suffix = '_待核' if b.get('suspect') else ''
        fn = f"{b['index']:03d}_{safe_name(b['title'])}{suffix}.pdf"
        nd = pymupdf.open()
        nd.insert_pdf(doc, from_page=s - 1, to_page=e - 1)
        nd.save(os.path.join(out_dir, fn), garbage=3, deflate=True)
        nd.close()
        produced.append(dict(file=fn, index=b['index'], start_phy=s, end_phy=e,
                             n_pages=e - s + 1, suspect=b.get('suspect', False),
                             with_front=b.get('with_front', False)))
    doc.close()
    sus_n = sum(1 for p in produced if p['suspect'])
    print(f'[3] 产出 {len(produced)} 个 PDF (含 {sus_n} 个待核) -> {out_dir}',
          flush=True)
    json.dump(dict(tag=tag, obs=obs, bounds=bounds, anomalies=anomalies,
                   produced=produced, total_pages=N, mode='fast'),
              open(os.path.join(PROBE, f'split_fast_result_{tag}.json'), 'w',
                   encoding='utf-8'), ensure_ascii=False, indent=1)
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
