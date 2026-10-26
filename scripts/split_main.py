"""
PDF 智能拆分 — 主流程 v2
=======================
关键改进（相对 v1）:
  1. 抽样加密 (步长 = N//60)，用 offset_fitter 拟合【分段 offset】
     —— 51 号存在 offset 从 5 跳到 7 的情况(中途插入 2 页)，全局单一 offset 会崩
  2. 锚点预测改用分段 offset
  3. 锚点验证失败时，在 ±search_win 页内搜索 (处理局部漂移/OCR 噪声)
  4. 输出异常清单 anomalies

用法: python split_main.py <tag> [--dry-run]
"""
import os, sys, json, gc, re
import numpy as np
from collections import Counter

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offset_fitter import fit_segments, predict_local

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
OUT_ROOT = r'F:\workBuddy\PDF智能拆分\output'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')


def safe_name(s, maxlen=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', '_', s or '')
    s = re.sub(r'\s+', '', s).strip(' .')
    return s[:maxlen] if s else 'untitled'


def main():
    tag = sys.argv[1]
    dry = '--dry-run' in sys.argv
    dpi = 300
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi') + 1])
    step_div = 60
    if '--dense' in sys.argv:
        step_div = 100

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

    src = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    doc = pymupdf.open(src); N = doc.page_count; doc.close()
    print(f'[{tag}] 总页={N}, 目录条目={len(entries)}', flush=True)

    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    # 若已存在全量页码映射(build_pnmap.py 产出), 优先查表免 OCR(大幅提速)
    pnmap = {}
    pnf = os.path.join(PROBE, f'pnmap_{tag}.json')
    use_pnmap = '--no-pnmap' not in sys.argv
    if use_pnmap and os.path.exists(pnf):
        try:
            pnmap = {int(k): v for k, v in
                     json.load(open(pnf, encoding='utf-8')).items()}
            print(f'[0] 命中全量页码映射 pnmap_{tag}.json ({len(pnmap)} 页)',
                  flush=True)
        except Exception:
            pnmap = {}

    cache = {}

    def read_pn(pi):          # pi: 0-based
        if pi in cache:
            return cache[pi]
        # 优先用 pnmap 查表(无需渲染 + OCR)
        if pnmap:
            v = pnmap.get(pi + 1)
            raw = str(v) if v is not None else ''
            cache[pi] = (v, raw)
            return v, raw
        arr = render_full(src, pi, dpi=dpi)
        loc = locate_page_number(arr)
        v, raw = None, ''
        if loc:
            im = crop_page_number(arr, loc['bbox'], scale=2)
            if im is not None:
                r2 = ocr.predict(np.array(im.convert('RGB')))
                txts = []
                for rr in r2:
                    txts += rr.get('rec_texts', [])
                raw = ''.join(txts).strip()
                v = int(raw) if re.fullmatch(r'\d{1,4}', raw) else None
        del arr; gc.collect()
        cache[pi] = (v, raw)
        return v, raw

    # ---- 1) 密集抽样 + 分段拟合 ----
    step = max(1, N // step_div)
    obs = []
    for pi in range(0, N, step):
        v, raw = read_pn(pi)
        if v is not None:
            obs.append((pi + 1, v))
    segs = fit_segments(obs)
    print(f'[1] 抽样 {len(obs)} 有效点, 拟合 {len(segs)} 段:', flush=True)
    for s in segs:
        print(f"      物理{s['start_phy']:>4}~{s['end_phy']:<4} offset={s['offset']:>3} "
              f"页码{s['page_lo']}~{s['page_hi']}", flush=True)
    # ---- 2) 锚点预测 + 验证(带窗口搜索) ----
    # 重要: 预测值(predict_local)本身是主依据; OCR 确认只是置信度标记。
    #       实测中 OCR 常把页脚英文(Add/EMAIL/ddr)误裁进来, 导致 got 非数字,
    #       但预测页仍然正确。故 OCR 未确认 -> 记 confirm=False, 不改锚点。
    # 遇错策略(用户拍板): 不再"遇错即停", 而是【标记异常 + 继续切分】。
    #       目录页码倒退的行, 其真实位置由"正文物理页连续性"兜底推定:
    #       start_phy = 前一条的 end_phy + 1 (见 Phase2)。
    SEARCH_WIN = 3
    raw_bounds, anomalies = [], []
    for e in entries:
        sp, ep = e['start_page'], e['end_page']
        pred, loc_off, refs = predict_local(obs, sp)
        if pred is None:
            anomalies.append(dict(type='NO_SEGMENT', index=e['index'],
                                  start=sp, title=e['title']))
            raw_bounds.append(dict(index=e['index'], title=e['title'],
                                   start_page=sp, end_page=ep,
                                   cand_phy=1, pred_phy=None, got_raw='',
                                   confirm=False, suspect=True))
            continue
        # 窗口搜索: 若预测页 OCR 不等于目录页码, 在 ±WIN 内找匹配
        found, found_raw = pred, ''
        v0, raw0 = read_pn(pred - 1) if 1 <= pred <= N else (None, '')
        found_raw = raw0
        if v0 != sp:
            for k in range(1, SEARCH_WIN + 1):
                hit = False
                for d in (-k, k):
                    cand = pred + d
                    if not (1 <= cand <= N):
                        continue
                    v, raw = read_pn(cand - 1)
                    if v == sp:
                        found, found_raw, hit = cand, raw, True
                        break
                if hit:
                    break
        confirm = (found_raw == str(sp))
        if found != pred:
            anomalies.append(dict(type='ANCHOR_SHIFT', index=e['index'], start=sp,
                                  pred=pred, found=found, title=e['title']))
        if not confirm:
            anomalies.append(dict(type='UNCONFIRMED', index=e['index'], start=sp,
                                  phy=found, got=found_raw, title=e['title']))
        raw_bounds.append(dict(index=e['index'], title=e['title'],
                               start_page=sp, end_page=ep, cand_phy=found,
                               pred_phy=pred, got_raw=found_raw, confirm=confirm,
                               suspect=False))
        mark = 'OK' if confirm else ('SHIFT' if found != pred else '?')
        print(f"  #{e['index']:>4} 页{sp:>4}->预测{pred:>4} 实锚{found:>4} "
              f"got={found_raw!r:>7} {mark}", flush=True)

    # ---- 3) 单调性修正(Phase2): 异常行处理 ----
    # 异常(目录页码倒退/物理页不递增)分为两类:
    #   A. 重复登记行: 该行目录页码 / 锚定物理页 与某条【正常行】完全相同
    #      -> 判定为纸质目录重复登记同一份实物, 标记 TOC_DUPLICATE_ROW 并【跳过不产出】
    #   B. 可修复行: 页码虽倒退但物理页可唯一确定
    #      -> 由"正文物理页连续性"兜底, 标记 _待核
    n_all = len(raw_bounds)
    # 判重: 若本行后面还存在一条【锚定物理页相同】的行, 则本行为重复登记。
    # 规则要点: 只向前看(后面的行优先保留), 避免 #12/#14 互相判重导致双双被跳过。
    #   实测依据: 孔隙93处的实物顶部标题是"混凝土试模", 与 #14 题名匹配 -> 保留 #14。
    for i, b in enumerate(raw_bounds):
        dup_of = None
        for j in range(i + 1, n_all):
            o = raw_bounds[j]
            if o.get('skip'):
                continue
            if o['cand_phy'] == b['cand_phy'] and b['cand_phy']:
                dup_of = o['index']
                break
        b['dup_of'] = dup_of
        if dup_of:
            b['skip'] = True
    # 顺序扫描: 用"前一条的最终物理页"判断, 保证连锁异常被一并识别
    # 重复登记行(skip=True, 上面已判定)不参与连续性判断, 直接标记跳过
    for b in raw_bounds:
        if b.get('skip'):
            b['fixed'] = False
            anomalies.append(dict(type='TOC_DUPLICATE_ROW', index=b['index'],
                                  start=b['start_page'], dup_of=b['dup_of'],
                                  title=b['title']))
            print(f"  x#{b['index']:>4} 目录重复登记(与 #{b['dup_of']} 同页 "
                  f"物理{b['cand_phy']}) -> 跳过不产出", flush=True)

    eff_prev_phy = None      # 前一条的最终物理页(跳过重复行)
    for i, b in enumerate(raw_bounds):
        if b.get('skip'):
            continue
        if eff_prev_phy is None:
            b['fixed'] = False
            eff_prev_phy = b['cand_phy']
            continue
        # 前一条"有效"行(跳过被跳过的行)
        prev = None
        for k in range(i - 1, -1, -1):
            if not raw_bounds[k].get('skip'):
                prev = raw_bounds[k]
                break
        backward = b['start_page'] < prev['end_page']
        nonmono = b['cand_phy'] <= eff_prev_phy
        b['fixed'] = bool(backward or nonmono)
        if b['fixed']:
            b['orig_phy'] = b['cand_phy']
            b['cand_phy'] = eff_prev_phy + 1     # 暂定, 后续段内均分再覆盖
            anomalies.append(dict(type='TOC_BACKWARD_FIXED', index=b['index'],
                                  start=b['start_page'], prev_end=prev['end_page'],
                                  title=b['title'], suspect=True))
            print(f"  !#{b['index']:>4} 目录页码异常({b['start_page']}, 前项末页{prev['end_page']})"
                  f" -> 改由正文连续性推定", flush=True)
        eff_prev_phy = b['cand_phy']

    # 找连续异常段, 用区间均分确定各自 start_phy
    #   P = 段前最后一条【有效】项的物理页; Q = 段后第一条【有效】项的物理页
    i = 0
    while i < n_all:
        if raw_bounds[i].get('skip') or not raw_bounds[i].get('fixed'):
            i += 1
            continue
        j = i
        while (j + 1 < n_all and raw_bounds[j + 1].get('fixed')
               and not raw_bounds[j + 1].get('skip')):
            j += 1
        # 向前找最近有效项
        P = 0
        for k in range(i - 1, -1, -1):
            if not raw_bounds[k].get('fixed') and not raw_bounds[k].get('skip'):
                P = raw_bounds[k]['cand_phy']
                break
        # 向后找最近有效项
        Q = N + 1
        for k in range(j + 1, n_all):
            if not raw_bounds[k].get('fixed') and not raw_bounds[k].get('skip'):
                Q = raw_bounds[k]['cand_phy']
                break
        cnt = j - i + 1
        span = max(cnt, Q - P - 1)          # 可用物理页数(至少 cnt 页)
        per = max(1, span // cnt)
        for k in range(cnt):
            b = raw_bounds[i + k]
            b['cand_phy'] = P + 1 + k * per
            b['suspect'] = True
            b['fixed_range'] = f'({P},{Q})/{cnt}条'
        print(f"  >> 异常段 #{raw_bounds[i]['index']}~#{raw_bounds[j]['index']}: "
              f"区间({P},{Q}) 共{cnt}条 -> 每条{per}页", flush=True)
        for k in range(cnt):
            b = raw_bounds[i + k]
            print(f"     ->#{b['index']:>4} 兜底锚定物理页 {b['cand_phy']}", flush=True)
        i = j + 1

    # 兜底后再做一次单调性清理(跳过被跳过的行)
    prev_eff = None
    for i in range(n_all):
        if raw_bounds[i].get('skip'):
            continue
        if prev_eff is not None and raw_bounds[i]['cand_phy'] <= prev_eff:
            raw_bounds[i]['cand_phy'] = prev_eff + 1
            raw_bounds[i]['suspect'] = True
        prev_eff = raw_bounds[i]['cand_phy']
    bounds = [b for b in raw_bounds if not b.get('skip')]

    # ---- 4) 计算 end_phy ----
    for b in bounds:
        b['start_phy'] = b['cand_phy']
    for i, b in enumerate(bounds):
        b['end_phy'] = (bounds[i + 1]['start_phy'] - 1
                        if i + 1 < len(bounds) else None)
    ok_n = sum(1 for b in bounds if b['confirm'])
    suspect_n = sum(1 for b in bounds if b['suspect'])
    dup_n = sum(1 for b in raw_bounds if b.get('skip'))
    print(f'\n[2] 边界={len(bounds)}(跳过{dup_n}), OCR确认={ok_n}, 存疑={suspect_n}, '
          f'异常={len(anomalies)}', flush=True)

    if dry:
        json.dump(dict(tag=tag, segments=segs, obs=obs, bounds=bounds,
                       anomalies=anomalies, total_pages=N),
                  open(os.path.join(PROBE, f'split_plan_{tag}.json'), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
        print('DRY_RUN_DONE', flush=True)
        return

    # ---- 5) 拆分 ----
    out_dir = os.path.join(OUT_ROOT, f'DHGS-GL5•1•1•TJ6•5-{tag}')
    os.makedirs(out_dir, exist_ok=True)
    # 末条边界: 目录给了区间(如 194~196 / 166~182)则用区间跨度推算;
    #           否则切到正文末尾(去掉可能的封底页, 即 N-1)
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
        # 需求: 封面 + 卷内目录 并入第一个拆分 PDF
        #       首个产出从物理第 1 页起切, 覆盖"封面 -> 目录 -> 第1条正文"整段
        if bi == 0 and s > 1:
            s = 1
            b['with_front'] = True
        if e < s:
            anomalies.append(dict(type='EMPTY_RANGE', index=b['index'])); continue
        # 存疑条目: 文件名加 _待核 后缀, 提示人工复核
        suffix = '_待核' if b.get('suspect') else ''
        fn = f"{b['index']:03d}_{safe_name(b['title'])}{suffix}.pdf"
        nd = pymupdf.open()
        nd.insert_pdf(doc, from_page=s - 1, to_page=e - 1)
        nd.save(os.path.join(out_dir, fn), garbage=3, deflate=True)
        nd.close()
        produced.append(dict(file=fn, index=b['index'], start_phy=s, end_phy=e,
                             n_pages=e - s + 1, suspect=b.get('suspect', False),
                             confirm=b.get('confirm', False),
                             with_front=b.get('with_front', False)))
    doc.close()
    sus_n = sum(1 for p in produced if p['suspect'])
    print(f'[3] 产出 {len(produced)} 个 PDF (含 {sus_n} 个待核) -> {out_dir}', flush=True)
    json.dump(dict(tag=tag, segments=segs, bounds=bounds, anomalies=anomalies,
                   produced=produced, total_pages=N),
              open(os.path.join(PROBE, f'split_result_{tag}.json'), 'w',
                   encoding='utf-8'), ensure_ascii=False, indent=1)
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
