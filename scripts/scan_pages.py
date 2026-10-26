"""页码全量识别（正式版）：定位 → OCR → 输出映射"""
import os, sys, json, re, traceback
import numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_map_engine import render_full, locate_page_number, crop_page_number
import pymupdf

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'

def scan_pdf(tag, limit=None, dpi=300, log=print):
    path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    doc = pymupdf.open(path); N = doc.page_count; doc.close()
    if limit: N = min(N, limit)

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    results = {}
    for pi in range(N):
        try:
            arr = render_full(path, pi, dpi=dpi)
            loc = locate_page_number(arr)
            if loc is None:
                results[pi+1] = dict(value=None, reason='NO_LOCATE')
                continue
            im = crop_page_number(arr, loc['bbox'], scale=2)
            if im is None:
                results[pi+1] = dict(value=None, reason='EMPTY_CROP')
                continue
            r2 = ocr.predict(np.array(im.convert('RGB')))
            txts = []
            for rr in r2:
                txts += rr.get('rec_texts', [])
            raw = ''.join(txts).strip()
            # 只接受纯数字
            v = int(raw) if re.fullmatch(r'\d{1,4}', raw) else None
            results[pi+1] = dict(value=v, raw=raw,
                                 relx=round(loc['relx'],3),
                                 rely=round(loc['rely'],4))
        except Exception as e:
            results[pi+1] = dict(value=None, reason=f'EXC {e!r}')
    return results, N


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else '51'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    LOG = open(f'_probe/scan2_{tag}.txt', 'w', encoding='utf-8')
    def log(*a):
        LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()
    try:
        res, N = scan_pdf(tag, limit, log=log)
        ok = sum(1 for v in res.values() if v['value'] is not None)
        log(f'{tag}: {N} 页, 识别到页码 {ok} 页 ({ok/N*100:.1f}%)')
        for k in sorted(res):
            log(f'  p{k:>4}: {res[k]}')
        json.dump(res, open(f'_probe/scan2_{tag}.json','w',encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        log('DONE_OK')
    except Exception as e:
        log('ERROR:', repr(e)); log(traceback.format_exc())
    finally:
        LOG.close()
