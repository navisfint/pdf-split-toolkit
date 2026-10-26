"""
页码垂直/水平位置分布测量
目的：验证"页码总是位于页面最底部条带、且靠左右两侧"这一前提
方法：对已知页码的页面，用 OCR 定位数字框，记录其归一化坐标
"""
import pymupdf, numpy as np
from PIL import Image
import io, os, traceback

PDF = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-51.pdf'
PDF67 = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-67.pdf'
LOG = open('_probe/pos_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

# 全页渲染（不裁剪），测量页码在整个页面中的归一化位置
SAMPLES = [
    (PDF,6,'51'),(PDF,7,'51'),(PDF,8,'51'),(PDF,9,'51'),(PDF,10,'51'),(PDF,11,'51'),(PDF,12,'51'),
    (PDF,51,'51'),(PDF,196,'51'),(PDF,202,'51'),
    (PDF67,4,'67'),(PDF67,5,'67'),(PDF67,10,'67'),(PDF67,41,'67'),(PDF67,91,'67'),(PDF67,121,'67'),
]

def render_full(path, pi, dpi=150):
    doc = pymupdf.open(path); pg = doc[pi-1]
    pix = pg.get_pixmap(dpi=dpi)
    arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
    doc.close(); return arr

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    log('ALL OCR boxes on FULL page (looking for the page number):')
    log('file  phys  truth | candidates (text, score, x0/W, y0/H, x1/W, y1/H)')
    log('-'*100)
    for path, pi, tag in SAMPLES:
        arr = render_full(path, pi)
        H, W = arr.shape[:2]
        p = f'_probe/ocr/full_{tag}_p{pi:03d}.png'
        Image.fromarray(arr).save(p)
        res = ocr.predict(p)
        items = []
        for r in res:
            texts = r.get('rec_texts', []); scores = r.get('rec_scores', [])
            boxes = r.get('rec_boxes', r.get('dt_polys', []))
            for t, s, b in zip(texts, scores, boxes):
                try:
                    b = np.array(b).reshape(-1,2)
                    x0,y0 = b[:,0].min(), b[:,1].min()
                    x1,y1 = b[:,0].max(), b[:,1].max()
                except Exception:
                    continue
                # 只关心纯数字且位于底部
                if t.strip().isdigit() and y0/H > 0.80:
                    items.append((t, round(float(s),3), round(x0/W,3), round(y0/H,3), round(x1/W,3), round(y1/H,3)))
        log(f'{tag}  p{pi:>4}  truth? | {items}')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
