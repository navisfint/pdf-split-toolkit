"""
探测文件边界 — 渲染指定物理页顶部，读标题行，判断是哪份文件。
用法: python probe_titles.py <tag> <phy_start> <phy_end> [step]
"""
import os, sys, io, gc
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
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dpi = 150

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    src = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    doc = pymupdf.open(src)
    print(f'物理页 {a}~{b}: 顶部标题 + 页码', flush=True)
    for pi in range(a - 1, min(b, doc.page_count), step):
        pix = doc[pi].get_pixmap(dpi=dpi)
        full = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        H = full.shape[0]
        # 顶部 35%
        top = full[:int(H * 0.35)]
        r2 = ocr.predict(top)
        txts = []
        for rr in r2:
            for t, s in zip(rr.get('rec_texts', []), rr.get('rec_scores', [])):
                if s > 0.6:
                    txts.append(t)
        # 底部页码区域 88%~100%
        bot = full[int(H * 0.86):]
        r3 = ocr.predict(bot)
        pn = []
        for rr in r3:
            for t, s in zip(rr.get('rec_texts', []), rr.get('rec_scores', [])):
                if s > 0.6:
                    pn.append(t)
        title = ' / '.join(txts[:3])[:70]
        print(f'  p{pi+1:>4}: [{title}]  底:{pn}', flush=True)
        del pix, full, top, bot, r2, r3
        gc.collect()
    doc.close()
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    main()
