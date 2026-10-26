"""测试目录解析器：对五份卷宗的目录页做解析并输出结构化结果"""
import os, sys, json, traceback
import numpy as np
import pymupdf
from PIL import Image
import io
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toc_parser import ocr_items, cluster_rows, extract_toc_rows

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
# 已验证的目录页范围 (1-based, 闭区间)
TOC = {'51': (2, 5), '60': (2, 3), '66': (2, 3), '67': (2, 3), '113': (2, 3)}

LOG = open('_probe/toc_parse_out.txt', 'w', encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')
    log('OCR ready')
    all_res = {}
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, (a, b) in TOC.items():
        if only and tag != only:
            continue
        try:
            path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
            doc = pymupdf.open(path)
            rows_all = []
            for i in range(a-1, b):
                pg = doc[i]
                pix = pg.get_pixmap(dpi=250)
                img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
                page_w = img.shape[1]
                res = ocr.predict(img)
                items = ocr_items(res)
                rows = cluster_rows(items, page_h=img.shape[0])
                toc_rows = extract_toc_rows(rows, page_w)
                for r in toc_rows:
                    r['src_page'] = i+1
                rows_all += toc_rows
                del pix, img, res, items, rows
            doc.close()
            all_res[tag] = rows_all
            log(f'===== {tag}: 解析出 {len(rows_all)} 行 =====')
            for r in rows_all:
                pg = r['page']
                pgs = f'{pg[0]}~{pg[1]}' if pg and pg[0]!=pg[1] else (str(pg[0]) if pg else 'None')
                log(f"  #{str(r['index']):>4} 页码={pgs:>10}  题名={r['title'][:44]}")
            log('')
            prev = {}
            if os.path.exists('_probe/toc_parse_result.json'):
                try:
                    prev = json.load(open('_probe/toc_parse_result.json', encoding='utf-8'))
                except Exception:
                    prev = {}
            prev.update(all_res)
            json.dump(prev, open('_probe/toc_parse_result.json','w',encoding='utf-8'),
                      ensure_ascii=False, indent=1)
        except Exception as e:
            log(f'!!!!! {tag} FAILED: {e!r}')
            log(traceback.format_exc())
    log('DONE_OK')
    json.dump(all_res, open(f'_probe/toc_parse_{only or "all"}.json','w',encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
