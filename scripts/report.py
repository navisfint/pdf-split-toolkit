"""
拆分结果汇总 — 生成人工复核清单
================================
读 _probe/split_result_*.json，输出:
  output/拆分报告.md     总览
  output/异常清单.csv    需人工复核的行

用法: python report.py
"""
import os, json, glob, csv

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_probe')
OUT = r'F:\workBuddy\PDF智能拆分\output'
TAG_NAME = {'51': 'DHGS-GL5•1•1•TJ6•5-51', '60': 'DHGS-GL5•1•1•TJ6•5-60',
            '66': 'DHGS-GL5•1•1•TJ6•5-66', '67': 'DHGS-GL5•1•1•TJ6•5-67',
            '113': 'DHGS-GL5•1•1•TJ6•5-113'}
TYPE_CN = {'TOC_DUPLICATE_ROW': '目录重复登记(无独立实物,已跳过)',
           'TOC_BACKWARD_FIXED': '目录页码异常(已按正文连续性推定)',
           'TOC_BACKWARD': '目录页码倒退',
           'UNCONFIRMED': 'OCR未确认(预测值可用)',
           'ANCHOR_SHIFT': '锚点窗口修正',
           'NO_SEGMENT': '无offset段', 'EMPTY_RANGE': '空区间'}


def main():
    lines = ['# PDF 智能拆分 — 结果报告', '']
    rows = []
    for p in sorted(glob.glob(os.path.join(PROBE, 'split_result_*.json'))):
        tag = os.path.basename(p).replace('split_result_', '').replace('.json', '')
        d = json.load(open(p, encoding='utf-8'))
        total = d['total_pages']
        bounds = d['bounds']
        prod = d.get('produced', [])
        anoms = d['anomalies']
        dup_ids = [a['index'] for a in anoms if a['type'] == 'TOC_DUPLICATE_ROW']
        toc_n = len(bounds) + len(dup_ids)
        lines.append(f'## {TAG_NAME.get(tag, tag)}')
        lines.append('')
        lines.append(f'- 总页数: **{total}**')
        lines.append(f'- 目录条目: {toc_n}')
        lines.append(f'- 产出 PDF: **{len(prod)}** 个')
        if dup_ids:
            lines.append(f'- ⏭️ 跳过重复登记行: {len(dup_ids)} 条 '
                         f'(#{"、#".join(str(i) for i in dup_ids)})')
        if prod:
            lines.append(f'- 覆盖物理页: {prod[0]["start_phy"]} ~ {prod[-1]["end_phy"]} '
                         f'({sum(x["n_pages"] for x in prod)} 页)')
        # offset 段
        segs = d.get('segments', [])
        if segs:
            lines.append(f'- 页码偏移分段: ' +
                         ', '.join(f'物理{s["start_phy"]}~{s["end_phy"]}(offset={s["offset"]})'
                                   for s in segs))
        lines.append('')
        # 异常
        if anoms:
            lines.append(f'### 异常 {len(anoms)} 条')
            lines.append('')
            lines.append('| 类型 | 序号 | 目录页码 | 预测物理页 | OCR实测 | 说明 |')
            lines.append('|---|---|---|---|---|---|')
            for a in anoms:
                rows.append(dict(卷宗=TAG_NAME.get(tag, tag), 类型=TYPE_CN.get(a['type'], a['type']),
                                 序号=a.get('index', ''), 目录页码=a.get('start', a.get('page', '')),
                                 预测物理页=a.get('pred', a.get('phy', '')),
                                 OCR实测=str(a.get('got', '')),
                                 说明=a.get('title', a.get('detail', ''))[:40]))
                lines.append(f"| {TYPE_CN.get(a['type'], a['type'])} | {a.get('index','')} | "
                             f"{a.get('start', a.get('page',''))} | {a.get('pred', a.get('phy',''))} | "
                             f"`{a.get('got','')}` | {a.get('title', a.get('detail',''))[:36]} |")
            lines.append('')
        else:
            lines.append('- ✅ 无异常')
            lines.append('')

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, '拆分报告.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    with open(os.path.join(OUT, '异常清单.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['卷宗', '类型', '序号', '目录页码', '预测物理页',
                                          'OCR实测', '说明'])
        w.writeheader()
        w.writerows(rows)
    print(f'报告已生成: {OUT}\\拆分报告.md')
    print(f'异常清单: {OUT}\\异常清单.csv ({len(rows)} 条)')


if __name__ == '__main__':
    main()
