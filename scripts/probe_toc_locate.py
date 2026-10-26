"""定位每份PDF的目录页：用 OCR 找"卷内目录"标题 + 表头关键词"""
import pymupdf, numpy as np, os, io, traceback
from PIL import Image
# 禁止联网下载模型，强制使用本地缓存
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
LOG = open('_probe/toc_locate.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

try:
    ocr = PaddleOCR(use_textline_orientation=False, lang='ch',
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    log('OCR ready')
    for tag in ['51','60','66','67','113']:
        path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
        doc = pymupdf.open(path)
        n = doc.page_count
        log(f'===== {tag} ({n}p) =====')
        for i in range(min(8, n)):
            pg = doc[i]
            # 只取页面上半部分(标题区)加速
            r = pg.rect
            clip = pymupdf.Rect(r.x0, r.y0, r.x1, r.y0+r.height*0.30)
            pix = pg.get_pixmap(dpi=150, clip=clip)
            img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
            res = ocr.predict(img)
            txts = []
            for rr in res:
                txts += rr.get('rec_texts', [])
            joined = ' | '.join(txts)
            hit = [k for k in ['卷内目录','序号','文件编号','责任者','文件题名','页号'] if k in joined]
            log(f'  p{i+1}: hits={hit}')
            log(f'      top_ocr={joined[:200]}')
        doc.close()
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
