"""
五份卷宗 · 目录页码列全量提取与异常统计
目的：验证"页码倒退"到底是普遍现象还是个别现象
"""
import pymupdf, numpy as np, os, io, re, json, traceback
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
LOG = open('_probe/toc_pages_stats.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

# 已验证的目录页范围(1-based)
TOC = {'51':(2,5), '60':(2,3), '66':(2,3), '67':(2,3), '113':(2,3)}

try:
    ocr = PaddleOCR(use_textline_orientation=True, lang='ch',
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    log('OCR ready')
    for tag,(a,b) in TOC.items():
        path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
        doc = pymupdf.open(path)
        log(f'===== {tag} toc p{a}~p{b} =====')
        for i in range(a-1, b):
            pg = doc[i]
            pix = pg.get_pixmap(dpi=200)
            img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
            res = ocr.predict(img)
            texts = []
            for rr in res:
                texts += rr.get('rec_texts', [])
            log(f'--- page {i+1} ---')
            log(' | '.join(texts))
        doc.close()
        log('')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
