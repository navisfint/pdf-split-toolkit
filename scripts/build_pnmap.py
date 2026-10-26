# -*- coding: utf-8 -*-
"""
构建 物理页 -> 正文页码 全量映射（一次扫描，落盘）
==============================================
输出 _probe/pnmap_{tag}.json:  {"1": null, "2": null, ..., "97": 93, ...}
null = 该页未识别到页码（插页或漏印）

用法: python build_pnmap.py <tag> <a> <b> [dpi]
"""
import os, sys, io, gc, json
import numpy as np
import pymupdf
from PIL import Image

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')


def main():
    tag = sys.argv[1]
    a, b = int(sys.argv[2]), int(sys.argv[3])
    dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 140

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    src = os.path.join(D, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}.pdf')
    doc = pymupdf.open(src)
    N = doc.page_count
    b = min(b, N)
    pnmap = {}
    for pi in range(a - 1, b):
        pix = doc[pi].get_pixmap(dpi=dpi)
        full = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        H = full.shape[0]
        bot = full[int(H * 0.86):]
        r3 = ocr.predict(bot)
        nums = []
        for rr in r3:
            for t, s in zip(rr.get('rec_texts', []), rr.get('rec_scores', [])):
                if s > 0.6 and t.strip().isdigit() and len(t.strip()) <= 4:
                    nums.append(int(t.strip()))
        # 取最右侧/最大的合理页码（>0）
        val = None
        for n in nums:
            if 0 < n <= 400:
                val = n if val is None else max(val, n)
        pnmap[str(pi + 1)] = val
        print(f'p{pi+1:>4}: {val}', flush=True)
        del pix, full, bot, r3
        gc.collect()
    doc.close()
    # 合并已有
    fp = os.path.join(PROBE, f'pnmap_{tag}.json')
    old = {}
    if os.path.exists(fp):
        old = json.load(open(fp, encoding='utf-8'))
    old.update(pnmap)
    json.dump(old, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print(f'SAVED {len(old)} entries -> {fp}', flush=True)
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
