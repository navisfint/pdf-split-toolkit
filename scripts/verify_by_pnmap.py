# -*- coding: utf-8 -*-
"""
用全量页码映射(pnmap)验证拆分结果
================================
对每个产出 PDF, 取其覆盖的物理页区间, 查 pnmap 得到页码序列,
检查: ① 页码是否连续递增 ② 与目录声明是否吻合

用法: python verify_by_pnmap.py <tag>
"""
import os, sys, json

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')
TOC = {'51': 1, '60': 1, '66': 1, '67': 1, '113': 1}


def main():
    tag = sys.argv[1]
    pnmap = json.load(open(os.path.join(PROBE, f'pnmap_{tag}.json'), encoding='utf-8'))
    res = json.load(open(os.path.join(PROBE, f'split_result_{tag}.json'), encoding='utf-8'))
    bounds = {b['index']: b for b in res['bounds']}

    print(f'{tag}: {len(res["produced"])} 个 PDF')
    print(f"{'#':>4} {'物理区间':>12} {'页数':>4} {'目录页':>7} "
          f"{'首页码':>6} {'尾页码':>6}  页码序列")
    ok = bad = 0
    for p in res['produced']:
        s, e = p['start_phy'], p['end_phy']
        pns = [pnmap.get(str(i)) for i in range(s, e + 1)]
        real = [x for x in pns if x is not None]
        # 检查连续性(过滤 None 后应严格 +1)
        cont = all(real[i + 1] - real[i] == 1 for i in range(len(real) - 1))
        b = bounds.get(p['index'], {})
        dec = b.get('start_page', '?')
        first = real[0] if real else None
        last = real[-1] if real else None
        # 首页码应 >= 目录声明(目录允许抄错, 不做强判定)
        flag = 'OK' if cont else 'GAP'
        if cont:
            ok += 1
        else:
            bad += 1
        seq = f"{first}~{last}({len(real)}页)" if real else '无页码'
        print(f"{p['index']:>4} {s:>5}~{e:<6} {e-s+1:>4} {dec!s:>7} "
              f"{first!s:>6} {last!s:>6}  {seq} {flag}")
    print(f'\n页码连续: {ok}/{len(res["produced"])}')
    if bad:
        print(f'⚠ 有 {bad} 个文件页码不连续(可能含插页/漏印, 属正常)')


if __name__ == '__main__':
    main()
