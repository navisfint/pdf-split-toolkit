"""
拆分质量抽查 — 验证切分边界的页码
==================================
对每份卷宗的产出 PDF，抽查首/尾页的页码，核对是否与目录声明一致。

用法: python verify_split.py <tag> [n_sample]
"""
import os, sys, json, io, re
import numpy as np
import pymupdf
from PIL import Image

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_map_engine import locate_page_number, crop_page_number
from paddleocr import PaddleOCR

OUT = r'F:\workBuddy\PDF智能拆分\output'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')


def render_page(doc, pi, dpi=300):
    pix = doc[pi].get_pixmap(dpi=dpi)
    return np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))


def main():
    tag = sys.argv[1]
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    plan = json.load(open(os.path.join(PROBE, f'split_result_{tag}.json'), encoding='utf-8'))
    prod = plan['produced']
    bounds = {b['index']: b for b in plan['bounds']}

    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    d = os.path.join(OUT, f'DHGS-GL5•1•1•TJ6•5-{tag}')
    step = max(1, len(prod) // n_sample)
    sample = prod[::step][:n_sample]
    print(f'{tag}: 抽查 {len(sample)}/{len(prod)} 个文件')
    print(f"{'序号':>5} {'目录页':>6} {'文件页数':>7} {'首页页码':>8} {'尾页页码':>8}  判定")
    ok = 0
    for p in sample:
        fp = os.path.join(d, p['file'])
        doc = pymupdf.open(fp)
        n = doc.page_count
        first = render_page(doc, 0)
        last = render_page(doc, n - 1)
        doc.close()

        def read(arr):
            loc = locate_page_number(arr)
            if not loc:
                return '-'
            im = crop_page_number(arr, loc['bbox'], scale=2)
            if im is None:
                return '-'
            r = ocr.predict(np.array(im.convert('RGB')))
            t = ''
            for rr in r:
                t += ''.join(rr.get('rec_texts', []))
            return t.strip() or '-'

        f_pn, l_pn = read(first), read(last)
        b = bounds.get(p['index'], {})
        dec = b.get('start_page', '?')
        good = (f_pn == str(dec))
        ok += good
        print(f"{p['index']:>5} {dec!s:>6} {n:>7} {f_pn:>8} {l_pn:>8}  "
              f"{'✓' if good else '✗'}")
    print(f'\n首页页码匹配: {ok}/{len(sample)}')


if __name__ == '__main__':
    main()
