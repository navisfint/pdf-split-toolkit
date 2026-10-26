"""
概念验证 v2：底部条带数字定位
改进点：
 1. 进一步收窄条带，只取最底部区域
 2. 先检测并移除长横线（形态学开运算）
 3. 连通域分析，放宽高度约束但加"数字宽高比 + 位置"约束
"""
import pymupdf, numpy as np
from PIL import Image
from scipy import ndimage
import io, os, traceback

PDF = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集\DHGS-GL5•1•1•TJ6•5-51.pdf'
OUT = '_probe/cc2'
os.makedirs(OUT, exist_ok=True)
LOG = open('_probe/cc2_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

SAMPLES = [(6,1),(7,2),(8,3),(9,4),(10,5),(11,6),(12,7),(51,46),(202,195),(196,189)]

d = pymupdf.open(PDF)

def render_bottom(pi, dpi=300, y_frac=0.90):
    pg = d[pi-1]; r = pg.rect
    clip = pymupdf.Rect(r.x0, r.y0+r.height*y_frac, r.x1, r.y1)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    return np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))

def remove_lines(bw):
    """移除长横线/竖线"""
    H, W = bw.shape
    # 横线：宽 > W*0.25 的连通域
    h_kernel = np.ones((1, max(20, W//12)), bool)
    horiz = ndimage.binary_opening(bw, structure=h_kernel)
    v_kernel = np.ones((max(20, H//8), 1), bool)
    vert = ndimage.binary_opening(bw, structure=v_kernel)
    return bw & ~(horiz | vert)

def find_digits(arr):
    """找数字块"""
    thr = np.quantile(arr, 0.12)
    bw = arr < thr
    bw = remove_lines(bw)
    lbl, n = ndimage.label(bw)
    H, W = arr.shape
    out = []
    for i in range(1, n+1):
        ys, xs = np.where(lbl==i)
        h = ys.max()-ys.min()+1; w = xs.max()-xs.min()+1
        area = len(xs)
        # 数字：高占条带 20%~92%，宽高比 0.15~1.0，填充率合理
        if not (H*0.18 <= h <= H*0.95): continue
        if not (0.12 <= w/h <= 1.05): continue
        if area < h*w*0.12: continue
        # 位置：左 30% 或 右 25%
        x0r = xs.min()/W
        if not (x0r < 0.32 or x0r > 0.68): continue
        out.append(dict(x0=int(xs.min()),x1=int(xs.max()),y0=int(ys.min()),
                        y1=int(ys.max()),h=int(h),w=int(w),area=int(area),rel=x0r))
    return out, bw

try:
    log('物理页 | 真值 | 候选块 (x0, h, w) | 数量')
    log('-'*72)
    for pi, truth in SAMPLES:
        arr = render_bottom(pi)
        digits, bw = find_digits(arr)
        digits.sort(key=lambda b: b['x0'])
        desc = [(b['x0'], b['h'], b['w']) for b in digits[:6]]
        log(f'p{pi:>4} | {truth:>4} | {desc} | n={len(digits)}')
        Image.fromarray(255-(bw*255).astype(np.uint8)).save(f'{OUT}/bw_p{pi:03d}.png')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', e); log(traceback.format_exc())
finally:
    LOG.close()
d.close()
