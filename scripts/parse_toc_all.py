"""
目录解析（正式版）— 逐页处理，逐份落盘，规避进程被杀。
用法: python parse_toc_all.py [tag ...]      # 不给 tag 则跑全部五份
输出: _probe/toc_{tag}.json  + 汇总 _probe/toc_all.json

关键修复（相对早期版本）：
  1. use_doc_orientation_classify=False —— 禁用文档方向分类。
     根因：PaddleOCR 会把"稀疏表格页"误判为旋转 180°，导致整页 OCR 坐标翻转、
     内容行跑到页面下方、序号/页码错位。禁用后 67号p3 恢复正常。
  2. 序号列阈值 0.135 -> 0.16
  3. 页码列改自适应"最右列"（不用 0.72 硬阈值，避免吞掉日期列 0.75~0.79）
  4. 页码支持跨行区间格式 "117~128"
"""
import os, sys, json, io, gc, traceback
import numpy as np
import pymupdf
from PIL import Image

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toc_parser import ocr_items, cluster_rows, extract_toc_rows

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
# 已验证的目录页范围 (1-based, 闭区间)
TOC = {'51': (2, 5), '60': (2, 3), '66': (2, 3), '67': (2, 3), '113': (2, 3)}
DPI = 250


def log(f, *a):
    f.write(' '.join(str(x) for x in a) + '\n'); f.flush()


def main():
    tags = sys.argv[1:] or list(TOC.keys())
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec')

    all_res = {}
    with open('_probe/toc_all_log.txt', 'w', encoding='utf-8') as f:
        log(f, 'OCR_READY')
        for tag in tags:
            if tag not in TOC:
                continue
            a, b = TOC[tag]
            try:
                path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
                doc = pymupdf.open(path)
                rows_all = []
                for i in range(a - 1, b):
                    pix = doc[i].get_pixmap(dpi=DPI)
                    img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
                    page_w = img.shape[1]
                    res = ocr.predict(img)
                    items = ocr_items(res)
                    rows = cluster_rows(items, page_h=img.shape[0])
                    for r in extract_toc_rows(rows, page_w):
                        r['src_page'] = i + 1
                        rows_all.append(r)
                    del pix, img, res, items, rows
                    gc.collect()
                doc.close()
                all_res[tag] = rows_all
                log(f, f'===== {tag}: {len(rows_all)} 行 =====')
                for r in rows_all:
                    pg = r['page']
                    pgs = f'{pg[0]}~{pg[1]}' if pg and pg[0] != pg[1] else (str(pg[0]) if pg else 'None')
                    log(f, f"  #{str(r['index']):>4} p{pgs:>10} raw={r['page_raw']!r:>10} {r['title'][:40]}")
                json.dump(all_res, open(f'_probe/toc_{tag}.json', 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
            except Exception as e:
                log(f, f'!!!!! {tag} FAILED: {e!r}')
                log(f, traceback.format_exc())
        json.dump(all_res, open('_probe/toc_all.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        log(f, 'DONE_OK')


if __name__ == '__main__':
    main()
