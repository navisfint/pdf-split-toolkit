"""
单页目录解析 — 一次只处理一页，规避进程被杀问题。
用法: python parse_one_page.py <tag> <page_1based> [dpi]
输出: _probe/parse_{tag}_p{page}.json  + 日志到 stdout(由调用方重定向)
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


def main():
    tag = sys.argv[1]
    page = int(sys.argv[2])
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
    doc = pymupdf.open(path)
    pg = doc[page - 1]
    pix = pg.get_pixmap(dpi=dpi)
    img = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))
    doc.close()
    del pg, pix
    gc.collect()
    print(f'RENDER_OK {tag} p{page} shape={img.shape}', flush=True)
    page_w = img.shape[1]

    # 落盘渲染图，便于人工核查
    Image.fromarray(img).save(f'_probe/{tag}_p{page}_render.png')

    from paddleocr import PaddleOCR
    kw = dict(use_textline_orientation=False,
              text_detection_model_name='PP-OCRv6_medium_det',
              text_recognition_model_name='PP-OCRv6_medium_rec')
    if len(sys.argv) > 4 and sys.argv[4] == 'nodocori':
        kw['use_doc_orientation_classify'] = False
        kw['use_doc_unwarping'] = False
    ocr = PaddleOCR(**kw)
    print('OCR_READY', flush=True)
    res = ocr.predict(img)
    items = ocr_items(res)
    print(f'OCR_ITEMS {len(items)}', flush=True)
    for it in sorted(items, key=lambda x: x['yc']):
        print(f"  yc={it['yc']:7.0f} yc/H={it['yc']/img.shape[0]:.3f} "
              f"xc/W={it['xc']/page_w:.3f} '{it['text']}'", flush=True)
    rows = cluster_rows(items, page_h=img.shape[0])
    toc_rows = extract_toc_rows(rows, page_w)
    print(f'PARSED {len(toc_rows)}', flush=True)
    for r in toc_rows:
        print(f"  #{r['index']} page={r['page']} raw='{r['page_raw']}' "
              f"title={r['title'][:50]}", flush=True)

    json.dump(dict(tag=tag, page=page, n_items=len(items),
                   rows=toc_rows,
                   raw=[{k: it[k] for k in ('text', 'score', 'xc', 'yc',
                                            'x0', 'y0', 'x1', 'y1')}
                        for it in items]),
              open(f'_probe/parse_{tag}_p{page}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('DONE_OK', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', repr(e), flush=True)
        traceback.print_exc()
