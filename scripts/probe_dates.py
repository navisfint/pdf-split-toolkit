# -*- coding: utf-8 -*-
"""读物理页底部日期+顶部日期"""
import os, sys, io, gc
import numpy as np, pymupdf
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'; os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
D = r'F:/工作/接单/kaggle/9.19+小柠+PDF/数据集'

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
        pix = doc[pi-1].get_pixmap(dpi=170)
        full = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
        H = full.shape[0]
        top = full[:int(H*0.40)]; bot = full[int(H*0.75):]
        r2 = ocr.predict(top); r3 = ocr.predict(bot)
        tt=[]; 
        for rr in r2:
            for t,s in zip(rr.get('rec_texts',[]), rr.get('rec_scores',[])):
                if s>0.5: tt.append(t)
        bb=[]
        for rr in r3:
            for t,s in zip(rr.get('rec_texts',[]), rr.get('rec_scores',[])):
                if s>0.5: bb.append(t)
        dev=[t for t in tt if '试模' in t]
        print(f'p{pi:>4}: dev={dev[:1]} | TOP: {" ".join(tt)[:55]}', flush=True)
        print(f'        BOT: {" ".join(bb)[:80]}', flush=True)
        del pix, full, top, bot, r2, r3; gc.collect()
    doc.close(); print('DONE_OK', flush=True)

main()
