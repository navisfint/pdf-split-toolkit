"""PaddleOCR 3.x 实测：识别底部条带页码"""
import pymupdf, numpy as np
from PIL import Image
import io, os, traceback

PDF = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-51.pdf'
PDF67 = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-67.pdf'
OUT = '_probe/ocr'
os.makedirs(OUT, exist_ok=True)
LOG = open('_probe/ocr_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

# (pdf, 物理页, 真值)
SAMPLES = [
    (PDF, 6, 1), (PDF, 7, 2), (PDF, 8, 3), (PDF, 9, 4), (PDF, 10, 5),
    (PDF, 11, 6), (PDF, 12, 7), (PDF, 51, 46), (PDF, 202, 195), (PDF, 196, 189),
    (PDF67, 4, 1), (PDF67, 5, 2), (PDF67, 10, 7), (PDF67, 41, 38),
    (PDF67, 91, 88), (PDF67, 121, 118),
]

def render_bottom(path, pi, dpi=300, y_frac=0.88):
    doc = pymupdf.open(path)
    pg = doc[pi-1]; r = pg.rect
    clip = pymupdf.Rect(r.x0, r.y0+r.height*y_frac, r.x1, r.y1)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
    doc.close()
    return arr

try:
    from paddleocr import PaddleOCR
    log('loading PaddleOCR...')
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    log('OCR loaded')
    log('页 | 真值 | OCR结果(text:score@x)')
    log('-'*80)
    for path, pi, truth in SAMPLES:
        tag = '51' if path==PDF else '67'
        arr = render_bottom(path, pi)
        p = f'{OUT}/{tag}_p{pi:03d}.png'
        Image.fromarray(arr).save(p)
        res = ocr.predict(p)
        items = []
        for r in res:
            texts = r.get('rec_texts', []); scores = r.get('rec_scores', [])
            boxes = r.get('rec_boxes', r.get('dt_polys', []))
            for t, s, b in zip(texts, scores, boxes):
                try:
                    x = int(b[0][0]) if hasattr(b[0],'__len__') else int(b[0])
                except Exception:
                    x = -1
                items.append((t, round(float(s),3), x))
        log(f'{tag} p{pi:>4} | 真值{truth:>4} | {items}')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
