"""
页码定位实验 v3
思路（定位与识别解耦）：
  1. 裁取页面底部区域 (y ∈ [0.85, 1.0])
  2. 二值化后做连通域分析
  3. 用"字号"特征筛出候选：高度显著大于同排文字
  4. 用"位置"特征筛出候选：靠左 (<30%宽) 或靠右 (>70%宽)
  5. 输出候选小图 + 坐标，交由 OCR 做单字符识别

本脚本只验证【第 3、4 步能否稳定框出页码】，识别部分单独做。
"""
import pymupdf, numpy as np, os, io, json, traceback
from PIL import Image
from scipy import ndimage

D = r'F:\工作\接单\kaggle\9.19+小柠+PDF\数据集'
OUT = '_probe/loc3'
os.makedirs(OUT, exist_ok=True)
LOG = open('_probe/loc3_out.txt','w',encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a)+'\n'); LOG.flush()

# (tag, 物理页, 真值) —— 覆盖已知样本
SAMPLES = [
 ('51',6,1),('51',7,2),('51',8,3),('51',9,4),('51',10,5),('51',12,7),
 ('51',196,189),('51',202,195),
 ('67',4,1),('67',5,2),('67',10,7),('67',41,38),('67',91,88),('67',121,118),
 ('60',4,1),('60',5,2),('60',185,181),
 ('113',4,1),('113',5,2),
]

def get_bottom(path, pi, y0f=0.84, dpi=300):
    doc = pymupdf.open(path); pg = doc[pi-1]; r = pg.rect
    clip = pymupdf.Rect(r.x0, r.y0+r.height*y0f, r.x1, r.y1)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    arr = np.array(Image.open(io.BytesIO(pix.tobytes('png'))).convert('L'))
    doc.close(); return arr

def locate(arr, debug_name=None):
    """返回候选块列表"""
    H, W = arr.shape
    # Otsu 类自适应阈值
    thr = int(np.quantile(arr, 0.10))
    bw = arr < thr
    lbl, n = ndimage.label(bw)
    # 统计所有连通域尺寸
    objs = ndimage.find_objects(lbl)
    comps = []
    for i, sl in enumerate(objs):
        if sl is None: continue
        ys, xs = sl
        h = ys.stop - ys.start; w = xs.stop - xs.start
        area = int((lbl[sl] == i+1).sum())
        comps.append(dict(idx=i+1, x0=int(xs.start), x1=int(xs.stop),
                          y0=int(ys.start), y1=int(ys.stop),
                          h=int(h), w=int(w), area=area,
                          yc=(ys.start+ys.stop)/2, xc=(xs.start+xs.stop)/2))
    # 取"高度"分布，找显著大的
    hs = np.array([c['h'] for c in comps])
    if len(hs) == 0: return [], bw, comps
    med = np.median(hs)
    # 页码特征：h >= 1.6*med, 且 w/h in [0.1, 1.2], 位于左右边缘
    cands = []
    for c in comps:
        if c['h'] < med*1.6: continue
        if not (0.08 <= c['w']/c['h'] <= 1.3): continue
        relx = c['xc']/W
        if not (relx < 0.30 or relx > 0.62): continue
        c['relx'] = round(relx,3); c['rely'] = round(c['yc']/H,3)
        c['ratio_h_med'] = round(c['h']/med,2)
        cands.append(c)
    return cands, bw, comps

try:
    for tag, pi, truth in SAMPLES:
        path = os.path.join(D, f'DHGS-GL5•1•1•TJ6•5-{tag}.pdf')
        arr = get_bottom(path, pi)
        H, W = arr.shape
        cands, bw, comps = locate(arr)
        cands.sort(key=lambda c: (-c['h']))
        desc = [(c['x0'], c['y0'], c['h'], c['w'], c['ratio_h_med'], c['relx']) for c in cands[:4]]
        log(f'{tag} p{pi:>4} 真值{truth:>4} | 连通域总数={len(comps)} 候选={len(cands)}')
        for d in desc:
            log(f'      x0={d[0]:>5} y0={d[1]:>4} h={d[2]:>4} w={d[3]:>3} h/med={d[4]:>5} relx={d[5]}')
        Image.fromarray(255-(bw*255).astype(np.uint8)).save(f'{OUT}/{tag}_p{pi:03d}.png')
    log('DONE_OK')
except Exception as e:
    log('ERROR:', repr(e)); log(traceback.format_exc())
finally:
    LOG.close()
