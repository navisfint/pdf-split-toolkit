# -*- coding: utf-8 -*-
"""快速探测物理页范围的底部页码(只读底部, 快)"""
import os, sys, io, gc
import numpy as np, pymupdf
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'; os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
D = r'F:/工作/接单/kaggle/9.19+小柠+PDF/数据集'

def main():
    tag = sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3])
    step = int(sys.argv[4]) if len(sys.argv)>4 else 1
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    doc = pymupdf.open(os.path.join(D, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}.pdf'))
    out=[]
    for pi in range(a-1, min(b, doc.page_count), step):
        pix = doc[pi].get_pixmap(dpi=140)
        full = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        H=full.shape[0]
        bot = full[int(H*0.86):]
        r3 = ocr.predict(bot); pn=[]
        for rr in r3:
            for t,s in zip(rr.get('rec_texts',[]), rr.get('rec_scores',[])):
                if s>0.6: pn.append(t)
        nums=[t for t in pn if t.strip().isdigit()]
        print(f'p{pi+1:>4}: {nums}', flush=True)
        del pix, full, bot, r3; gc.collect()
    doc.close(); print('DONE_OK', flush=True)

main()
