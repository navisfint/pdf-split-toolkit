"""
结构探测：自动判定卷宗的 封面 / 目录页范围 / 正文起点
依据：目录页含特殊表格线(竖线多) + 标题"卷内目录"；正文页右下有页码
策略（不依赖OCR，纯图像统计 + 少量OCR确认）：
  1) 用页面横向长横线数量区分 目录页(表格,横线多) 与 正文页(横线少)
  2) 首页(封面)通常是牛皮纸底、无表格线
"""
import pymupdf, numpy as np, os, io, json, traceback
from PIL import Image

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
LOG = open('_probe/struct_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

def page_lines(path, pi, dpi=120):
    """返回 (横向长线条数, 竖线长线条数, 平均灰度)"""
    doc = pymupdf.open(path); pg = doc[pi-1]
    pix = pg.get_pixmap(dpi=dpi)
    arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))
    doc.close()
    H, W = arr.shape
    bw = arr < 160
    # 横向: 每行黑点占比 > 60% 视为横线
    row_ratio = bw.sum(axis=1) / W
    hlines = int((row_ratio > 0.60).sum())
    # 竖线: 每列黑点占比 > 50%
    col_ratio = bw.sum(axis=0) / H
    vlines = int((col_ratio > 0.50).sum())
    return hlines, vlines, float(arr.mean())

try:
    for tag in ['51','60','66','67','113']:
        path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
        doc = pymupdf.open(path)
        n = doc.page_count
        log(f'===== {tag}  pages={n} =====')
        for i in range(1, min(10, n)+1):
            h, v, g = page_lines(path, i)
            flag = ''
            if v >= 3 and h >= 3: flag = '<-- 表格页(疑似目录)'
            if h <= 1 and v <= 1: flag = '<-- 无表格线'
            log(f'  p{i:>3}: hlines={h:>4} vlines={v:>4} gray={g:>6.1f} {flag}')
        doc.close()
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
