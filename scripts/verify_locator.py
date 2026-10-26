"""页码定位器验证：定位 → 裁剪 → OCR 单字符识别 → 与真值对比"""
import os, sys, traceback, numpy as np
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from page_number_locator import render_full, locate_page_number_full, crop_digits

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
OUT = '_probe/verify_loc'
os.makedirs(OUT, exist_ok=True)
LOG = open('_probe/verify_loc_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

SAMPLES = [
 ('51',6,1),('51',7,2),('51',8,3),('51',9,4),('51',10,5),('51',11,6),('51',12,7),
 ('51',51,46),('51',196,189),('51',202,195),
 ('67',4,1),('67',5,2),('67',10,7),('67',41,38),('67',91,88),('67',121,118),
 ('60',4,1),('60',5,2),('60',100,97),('60',185,181),
 ('113',4,1),('113',5,2),('113',60,56),
]

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, lang='en',
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    log('OCR ready')
    log(f"{'卷':>4} {'页':>5} {'真值':>5} | {'定位':<28} | OCR结果")
    log('-'*88)
    ok = 0; total = 0
    for tag, pi, truth in SAMPLES:
        total += 1
        path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
        try:
            arr = render_full(path, pi-1, dpi=200)
            res = locate_page_number_full(arr)
            if res is None:
                log(f'{tag:>4} {pi:>5} {truth:>5} | 未定位到')
                continue
            x0,y0,x1,y1 = res['bbox']
            nblocks = len(res['digits'])
            crop = crop_digits(arr, res['bbox'])
            cp = f'{OUT}/{tag}_p{pi:03d}.png'
            Image.fromarray(crop).save(cp)
            # OCR 识别
            r2 = ocr.predict(cp)
            txts = []
            for rr in r2:
                txts += rr.get('rec_texts', [])
            got = ''.join(txts).strip()
            mark = 'OK' if got == str(truth) else 'XX'
            if got == str(truth): ok += 1
            log(f'{tag:>4} {pi:>5} {truth:>5} | bbox=({x0},{y0},{x1},{y1}) n={nblocks} | "{got}" [{mark}]')
        except Exception as e:
            log(f'{tag:>4} {pi:>5} {truth:>5} | EXC {e!r}')
    log(f'\n准确率: {ok}/{total}')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
