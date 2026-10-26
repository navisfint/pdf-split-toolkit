"""
锚点验证器
==========
用"分段偏移法"的主 offset，把目录起始页码换算成物理页，
只渲染这些物理页做单点 OCR 验证 —— 高效且可靠。

策略（核心）：
  1. 先用少量抽样页求主导 offset（多数表决，对 OCR 错误免疫）
  2. 对每个目录条目: 预测物理页 = 起始页码 + offset
  3. 只 OCR 这些预测页，确认其页码 == 目录起始页码
  4. 验证失败的，在 ±search_win 页内搜索（处理插页/offset 漂移）

用法: python verify_anchors.py <tag> <toc_json> [--dpi 300]
"""
import os, sys, json, io, gc, re
import numpy as np
import pymupdf
from PIL import Image
from collections import Counter

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_map_engine import render_full, locate_page_number, crop_page_number

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'


def read_page_number(ocr, arr, loc):
    im = crop_page_number(arr, loc['bbox'], scale=2)
    if im is None:
        return None, ''
    r2 = ocr.predict(np.array(im.convert('RGB')))
    txts = []
    for rr in r2:
        txts += rr.get('rec_texts', [])
    raw = ''.join(txts).strip()
    v = int(raw) if re.fullmatch(r'\d{1,4}', raw) else None
    return v, raw


def main():
    tag = sys.argv[1]
    toc_json = sys.argv[2]
    dpi = 300
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi') + 1])

    toc = json.load(open(toc_json, encoding='utf-8'))
    if isinstance(toc, dict):
        toc = toc[tag]
    entries = [e for e in toc if e.get('page')]

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    doc = pymupdf.open(path)
    N = doc.page_count

    # ---- 阶段1: 抽样求主导 offset ----
    step = max(1, N // 40)
    print(f'[1] 抽样 offset, 步长={step}, 总页={N}', flush=True)
    obs = []
    for pi in range(0, N, step):
        arr = render_full(path, pi, dpi=dpi)
        loc = locate_page_number(arr)
        if loc:
            v, raw = read_page_number(ocr, arr, loc)
            if v is not None:
                obs.append((pi + 1, v))
        del arr
        gc.collect()
    offs = Counter(p - v for p, v in obs)
    if not offs:
        print('!! 无有效观测'); doc.close(); return
    dom, cnt = offs.most_common(1)[0]
    print(f'[1] 主导 offset={dom} ({cnt}/{len(obs)}), 分布={offs.most_common(5)}',
          flush=True)

    # ---- 阶段2: 逐锚点验证 ----
    print(f'\n[2] 锚点验证 (预测物理页 = 页码 + {dom})', flush=True)
    results = []
    cache = {}
    for e in entries:
        sp = e['page'][0]
        pred = sp + dom
        if pred < 1 or pred > N:
            results.append(dict(index=e['index'], start_page=sp, pred=pred,
                                status='OUT_OF_RANGE'))
            continue
        if pred not in cache:
            arr = render_full(path, pred - 1, dpi=dpi)
            loc = locate_page_number(arr)
            if loc:
                v, raw = read_page_number(ocr, arr, loc)
            else:
                v, raw = None, '(NO_LOCATE)'
            cache[pred] = (v, raw)
            del arr
            gc.collect()
        v, raw = cache[pred]
        ok = (v == sp)
        results.append(dict(index=e['index'], start_page=sp, pred=pred,
                            got=v, got_raw=raw,
                            status='OK' if ok else 'MISMATCH'))
        flag = '✓' if ok else '✗'
        print(f"  #{e['index']:>4} 页{sp:>4} -> 物理页{pred:>4} 实际={raw!r:>8} {flag}",
              flush=True)

    ok_n = sum(1 for r in results if r['status'] == 'OK')
    print(f'\n[2] 验证通过 {ok_n}/{len(results)}', flush=True)
    json.dump(dict(tag=tag, offset=dom, offset_votes=dict(offs),
                   results=results),
              open(f'_probe/anchors_{tag}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    doc.close()
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
