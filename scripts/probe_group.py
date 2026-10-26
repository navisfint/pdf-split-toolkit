# -*- coding: utf-8 -*-
"""探测物理页的顶部标题关键词(判断属于哪台设备)"""
import os, sys, io, gc
import numpy as np, pymupdf
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'; os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
D = r'F:/工作/接单/kaggle/9.19+小柠+PDF/数据集'
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')

def main():
    tag = sys.argv[1]
    pages = [int(x) for x in sys.argv[2:]]
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    doc = pymupdf.open(os.path.join(D, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}.pdf'))
    for pi in pages:
        pix = doc[pi-1].get_pixmap(dpi=160)
        full = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        H = full.shape[0]
        top = full[:int(H*0.30)]
        r2 = ocr.predict(top)
        txts=[]
        for rr in r2:
            for t,s in zip(rr.get('rec_texts',[]), rr.get('rec_scores',[])):
                if s>0.6: txts.append(t)
        # 提取设备名关键词
        joined=' '.join(txts)
        dev=''
        for k in ['砂浆试模','混凝土试模','混凝土坑渗试模','坍落度筒']:
            if k in joined: dev=k; break
        print(f'p{pi:>4}: {dev}   | {joined[:60]}', flush=True)
        del pix, full, top, r2; gc.collect()
    doc.close(); print('DONE_OK', flush=True)

main()
