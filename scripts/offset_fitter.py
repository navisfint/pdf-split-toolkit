"""
分段 offset 拟合器
==================
输入: 抽样观测 [(物理页, 页码)] + 目录锚点 [(目录页码)]
输出: 分段 offset 表 + 每个锚点的物理页预测

核心思想:
  页码序列本质是"分段线性": offset = 物理页 - 页码 在一段内恒定，
  段与段之间因"插页"(无页码的页)产生跳变。

方法:
  1. 对观测点按物理页排序，计算相邻点的 offset 差值
  2. 用"局部众数"平滑: 每个点用邻近 k 个点的 offset 众数代替 (抗 OCR 噪声)
  3. 检测平滑后 offset 的跳变点 -> 分段
  4. 每个锚点: 找"目录页码落在该段页码范围内"的段，用该段 offset 预测
  5. 若某锚点在所有段都找不到 -> 用最近段的 offset 外推
"""
import json
from collections import Counter


def local_mode(seq, k=5):
    """滑动窗口众数平滑"""
    n = len(seq)
    out = []
    for i in range(n):
        lo = max(0, i - k // 2)
        hi = min(n, i + k // 2 + 1)
        vals = [seq[j] for j in range(lo, hi)]
        out.append(Counter(vals).most_common(1)[0][0])
    return out


def predict_local(obs, toc_page, k=8):
    """
    【推荐】局部最近邻 offset 预测。
    对目录页码 toc_page，找 obs 中页码最接近的 k 个点，取其 offset 众数作为局部 offset。
    天然处理"分段 offset"，也不依赖脆弱的段拟合。

    obs: [(phy, page), ...]
    返回 (预测物理页, 局部offset, 参考点) 或 (None, None, None)
    """
    if not obs:
        return None, None, None
    pts = sorted(obs, key=lambda x: abs(x[1] - toc_page))
    near = pts[:k]
    # 加权众数: 页码越近权重越高
    weights = Counter()
    for phy, pg in near:
        d = abs(pg - toc_page)
        w = max(1, 10 - d)
        weights[phy - pg] += w
    off = weights.most_common(1)[0][0]
    return toc_page + off, off, near[:3]


def fit_segments(obs, min_seg=4, min_page_span=3):
    """
    obs: [(phy, page), ...]  已按 phy 排序，page 非空
    返回 segments: [dict(start_phy, end_phy, offset, page_lo, page_hi, n_pts)]

    过滤规则（关键，抗 OCR 误识）:
      - 段的最少观测点数 >= min_seg
      - 段的页码跨度 >= min_page_span （否则是孤立异常点）
      - 与相邻段 offset 差值 <= 2 的段合并（同一物理段的小抖动）
    """
    if not obs:
        return []
    obs = sorted(obs)
    offs = [p - v for p, v in obs]
    smooth = local_mode(offs, k=5)

    raw_segs = []
    s_i = 0
    for i in range(1, len(smooth)):
        if smooth[i] != smooth[s_i]:
            raw_segs.append((s_i, i - 1, smooth[s_i]))
            s_i = i
    raw_segs.append((s_i, len(smooth) - 1, smooth[s_i]))

    # 构造段对象
    cand = []
    for a, b, off in raw_segs:
        cand.append(dict(start_phy=obs[a][0], end_phy=obs[b][0], offset=off,
                         page_lo=obs[a][1], page_hi=obs[b][1],
                         n_pts=b - a + 1))

    # 过滤: 点数太少 或 页码跨度太小 -> 丢弃(视为 OCR 噪声)
    kept = [s for s in cand
            if s['n_pts'] >= min_seg and (s['page_hi'] - s['page_lo']) >= min_page_span]
    if not kept:
        # 全被过滤时，退化为单一全局 offset（取众数）
        all_off = Counter(offs).most_common(1)[0][0]
        return [dict(start_phy=obs[0][0], end_phy=obs[-1][0], offset=all_off,
                     page_lo=obs[0][1], page_hi=obs[-1][1], n_pts=len(obs))]

    # 合并 offset 接近(<=2)的相邻段
    merged = [kept[0]]
    for s in kept[1:]:
        if abs(s['offset'] - merged[-1]['offset']) <= 2:
            merged[-1]['end_phy'] = s['end_phy']
            merged[-1]['page_hi'] = s['page_hi']
            merged[-1]['n_pts'] += s['n_pts']
        else:
            merged.append(s)
    return merged


def predict(segments, toc_page, total_pages=100000):
    """
    给目录页码，预测物理页。
    策略: 选出"页码区间覆盖 toc_page 且 n_pts 最多"的段；没有则用 n_pts 最多的段外推。
    """
    if not segments:
        return None
    best = None
    for s in segments:
        lo, hi = s['page_lo'] - 3, s['page_hi'] + 3
        if lo <= toc_page <= hi:
            if best is None or s['n_pts'] > best['n_pts']:
                best = s
    if best is None:
        best = max(segments, key=lambda s: s['n_pts'])
    return toc_page + best['offset'], best


def fit_from_scan(scan_json_or_list):
    """从扫描结果构造 obs"""
    if isinstance(scan_json_or_list, str):
        d = json.load(open(scan_json_or_list, encoding='utf-8'))
    else:
        d = scan_json_or_list
    obs = []
    for k, v in d.items():
        if isinstance(v, dict) and v.get('value') is not None:
            obs.append((int(k), int(v['value'])))
    return sorted(obs)


if __name__ == '__main__':
    import sys
    obs = fit_from_scan(sys.argv[1])
    segs = fit_segments(obs)
    print('观测', len(obs), '段数', len(segs))
    for s in segs:
        print(f"  物理{s['start_phy']:>4}~{s['end_phy']:<4} "
              f"offset={s['offset']:>3}  页码{s['page_lo']}~{s['page_hi']}")
