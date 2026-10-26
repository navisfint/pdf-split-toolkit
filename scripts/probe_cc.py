"""
概念验证：底部条带数字定位 + 连通域分析
目标：验证"打印体页码"能否用纯图像方法稳定定位（不依赖通用OCR）
"""
import pymupdf, numpy as np
from PIL import Image
import io, os, sys, traceback

PDF = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-51.pdf'
OUT = '_probe/cc'
os.makedirs(OUT, exist_ok=True)
LOG = open('_probe/cc_out.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.write(s + '\n'); LOG.flush()

# 已人工核实的样本: (物理页1-based, 页码真值)
SAMPLES = [(6,1),(7,2),(8,3),(9,4),(10,5),(11,6),(12,7),(51,46),(202,195),(196,189)]

d = pymupdf.open(PDF)

def render_bottom(pi, dpi=300, frac=0.14):
    pg = d[pi-1]
    r = pg.rect
    clip = pymupdf.Rect(r.x0, r.y0 + r.height*(1-frac), r.x1, r.y1)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    img = Image.open(io.BytesIO(pix.tobytes('png'))).convert('L')
    return np.array(img)

def find_cc(arr, min_h_ratio=0.25, max_h_ratio=0.85, dark_q=0.25):
    """连通域分析,找字符块"""
    from scipy import ndimage
    # 自适应二值化
    thr = np.quantile(arr, dark_q)
    bw = arr < thr
    lbl, n = ndimage.label(bw)
    boxes = []
    H = arr.shape[0]
    for i in range(1, n+1):
        ys, xs = np.where(lbl == i)
        h = ys.max()-ys.min()+1
        w = xs.max()-xs.min()+1
        if h < H*min_h_ratio or h > H*max_h_ratio: continue
        if w > h*1.2 or w < 2: continue   # 数字近似方形/竖长
        boxes.append({
            'x0':int(xs.min()),'x1':int(xs.max()),
            'y0':int(ys.min()),'y1':int(ys.max()),
            'h':int(h),'w':int(w),'area':int(bw[ys,xs].sum())
        })
    return boxes, bw

log('页 | 真值 | 定位到的数字块(x,h) | 数量')
log('-'*70)
try:
    for pi, truth in SAMPLES:
        arr = render_bottom(pi)
        H, W = arr.shape
        boxes, bw = find_cc(arr)
        cand = [b for b in boxes if b['x0'] < W*0.30 or b['x0'] > W*0.65]
        if cand:
            medh = np.median([b['h'] for b in cand])
            big = [b for b in cand if b['h'] >= medh*0.8]
        else:
            big = []
        desc = [(b['x0'],b['h']) for b in sorted(big,key=lambda b:-b['h'])[:5]]
        log(f'p{pi:>4} | {truth:>4} | {desc} | {len(big)}')
        Image.fromarray(255-(bw*255).astype(np.uint8)).save(f'{OUT}/bw_p{pi:03d}.png')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', e)
    log(traceback.format_exc())
finally:
    LOG.close()

d.close()
