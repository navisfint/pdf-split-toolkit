"""
探测指定物理页范围的页码 — 用于核对目录错误行的真实位置。
用法: python probe_range.py <tag> <phy_start> <phy_end> [step]
"""
import os, sys, io, re, gc
import numpy as np
import pymupdf
from PIL import Image

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_map_engine import render_full, locate_page_number, crop_page_number

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')


def main():
    tag = sys.argv[1]
    a = int(sys.argv[2]); b = int(sys.argv[3])
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dpi = 300

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    src = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    print(f'物理页 {a}~{b} (step={step})', flush=True)
    for pi in range(a - 1, b, step):
        arr = render_full(src, pi, dpi=dpi)
        loc = locate_page_number(arr)
        raw = '-'
        if loc:
            im = crop_page_number(arr, loc['bbox'], scale=2)
            if im is not None:
                r2 = ocr.predict(np.array(im.convert('RGB')))
                t = ''
                for rr in r2:
                    t += ''.join(rr.get('rec_texts', []))
                raw = t.strip() or '(空)'
        print(f'  p{pi+1:>4}: {raw}', flush=True)
        del arr; gc.collect()
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
