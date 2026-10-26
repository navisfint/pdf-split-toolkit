"""全量扫描 51 号，统计页码定位覆盖率"""
import os, sys, traceback, numpy as np
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_number_locator import render_full
from page_number_engine import locate_candidates, group_digits

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
TAG = sys.argv[1] if len(sys.argv) > 1 else '51'
OUT = f'_probe/scan_{TAG}'
os.makedirs(OUT, exist_ok=True)
LOG = open(f'_probe/scan_{TAG}_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{TAG}.pdf')
import pymupdf
doc = pymupdf.open(path)
N = doc.page_count
doc.close()
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else N
N = min(N, LIMIT)

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, lang='en',
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    log(f'{TAG}: {N} pages. OCR ready')
    hits = 0
    results = {}
    for pi in range(N):
        try:
            arr = render_full(path, pi, dpi=200)
            cands = locate_candidates(arr)
            if not cands:
                results[pi+1] = None
                continue
            grp = group_digits(cands)
            x0 = min(c['x0'] for c in grp); x1 = max(c['x1'] for c in grp)
            y0 = min(c['y0'] for c in grp); y1 = max(c['y1'] for c in grp)
            pad = 14
            H, W = arr.shape
            x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(W,x1+pad); y1=min(H,y1+pad)
            crop = arr[y0:y1, x0:x1]
            # 放大 2x 提升 OCR 精度
            im = Image.fromarray(crop).resize(
                (crop.shape[1]*2, crop.shape[0]*2), Image.LANCZOS)
            cp = f'{OUT}/p{pi+1:04d}.png'
            im.save(cp)
            r2 = ocr.predict(cp)
            txts = []
            for rr in r2:
                txts += rr.get('rec_texts', [])
            got = ''.join(txts).strip()
            results[pi+1] = got
            hits += 1
        except Exception as e:
            log(f'  p{pi+1} EXC {e!r}')
            results[pi+1] = None
    log(f'定位成功: {hits}/{N}')
    # 输出全部结果
    for k in sorted(results):
        log(f'  p{k:>4}: {results[k]!r}')
    import json
    json.dump(results, open(f'_probe/scan_{TAG}_result.json','w',encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
