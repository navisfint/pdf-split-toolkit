"""
页码列单调性分析：把五份目录的页码列人工抄录进来做校验
（因为 OCR 的文本顺序不保证行序，自动行绑定不可靠；此处用人工核对过的数据）
"""
import json

# 从 OCR 全文人工整理的 (序号, 页码原值) —— 已逐页核对
TOC_PAGES = {
 '51': [(1,1),(2,7),(3,10),(4,13),(5,16),(6,19),(7,22),(8,25),(9,37),(10,43),
        (11,49),(12,51),(13,57),(14,60),(15,63),(16,67),(17,70),(18,73),(19,76),(20,79),
        (21,82),(22,85),(23,88),(24,91),(25,94),(26,100),(27,103),(28,106),(29,107),(30,110),
        (31,113),(32,116),(33,119),(34,122),(35,125),(36,128),(37,131),(38,134),(39,137),(40,152),
        (41,158),(42,161),(43,164),(44,167),(45,170),(46,173),(47,176),(48,179),(49,182),(50,185),
        (51,188),(52,181),(53,'194~196')],
 '60': [(1,1),(2,13),(3,25),(4,37),(5,49),(6,61),(7,64),(8,70),(9,76),(10,82),
        (11,88),(12,93),(13,111),(14,93),(15,111),(16,129),(17,147),(18,'166~182')],
 '66': [(1,1),(2,13),(3,25),(4,31),(5,43),(6,55),(7,61),(8,73),(9,85),(10,91),
        (11,97),(12,109),(13,121),(14,127),(15,133),(16,139),(17,'146~151')],
 '67': [(1,1),(2,4),(3,9),(4,14),(5,20),(6,25),(7,31),(8,37),(9,41),(10,47),
        (11,58),(12,65),(13,72),(14,80),(15,87),(16,'117~128')],
 '113':[(1,1),(2,7),(3,12),(4,22),(5,32),(6,37),(7,42),(8,47),(9,52),(10,58),
        (11,68),(12,79),(13,90),(14,95),(15,100)],
}

# 各卷宗总页数与前置页数（已实测）
META = {
 '51': dict(total=204, offset=5),
 '60': dict(total=187, offset=4),
 '66': dict(total=155, offset=4),
 '67': dict(total=132, offset=4),
 '113':dict(total=125, offset=4),
}

def start_of(v):
    if isinstance(v, str):
        return int(v.split('~')[0])
    return v

def end_of(v, nxt):
    if isinstance(v, str):
        return int(v.split('~')[1])
    return start_of(nxt) - 1 if nxt else None

report = []
for tag, rows in TOC_PAGES.items():
    m = META[tag]
    body = m['total'] - m['offset']
    starts = [start_of(v) for _, v in rows]
    # 单调性检查
    bad = []
    for i in range(1, len(starts)):
        if starts[i] <= starts[i-1]:
            bad.append((rows[i-1][0], starts[i-1], rows[i][0], starts[i]))
    total_last = rows[-1][1]
    tail = end_of(rows[-1][1], None)
    report.append(dict(
        tag=tag, total=m['total'], body_pages=body, offset=m['offset'],
        n_items=len(rows), first=starts[0], last_start=starts[-1],
        last_end=tail, non_monotonic=bad,
        fits_body=(tail == body) if tail else None
    ))

LOG = open('_probe/monotonic_report.txt','w',encoding='utf-8')
for r in report:
    LOG.write(f"===== {r['tag']} =====\n")
    LOG.write(f"  总页={r['total']}  前置={r['offset']}  正文={r['body_pages']}  条目={r['n_items']}\n")
    LOG.write(f"  首个起始={r['first']}  末个起始={r['last_start']}  末个结束={r['last_end']}\n")
    LOG.write(f"  末个结束 == 正文页数 ? {r['fits_body']}\n")
    if r['non_monotonic']:
        LOG.write(f"  [!] 目录页码倒退 {len(r['non_monotonic'])} 处:\n")
        for prev_sn, prev_v, cur_sn, cur_v in r['non_monotonic']:
            LOG.write(f"      序号{prev_sn}(={prev_v})  ->  序号{cur_sn}(={cur_v})\n")
    else:
        LOG.write("  [OK] 页码严格递增\n")
    LOG.write("\n")
LOG.close()
print('done')
