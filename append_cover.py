# -*- coding: utf-8 -*-
"""
封皮追加工具（附属程序）
========================
背景
----
PDF 智能拆分主程序把档案卷宗按「卷内目录」切成 001.pdf、002.pdf …
但源 PDF 的**最后一页**（牛皮纸颜色的「卷内备考表」，属封底性质）不在目录条目里，
因此没被切进任何结果文件 —— 客户要求把它补进**最后一个**拆分 PDF 的末尾。

本程序独立于主程序，不修改 `core_split.py` 的任何逻辑，只做「事后补页」。

配对规则
--------
主程序产出结构：`<输出目录>/<源PDF文件名>/001.pdf …`
所以只要用子文件夹名 + '.pdf' 去「源 PDF 目录」找同名文件即可自动配对。

用法
----
    python append_cover.py --src <源PDF目录> --out <拆分结果目录> [--dry-run] [--backup]

    --dry-run  只打印将要做什么，不写文件
    --backup   改写前把原目标 PDF 备份为 xxx.bak.pdf（默认开）

幂等保护
--------
重复运行不会把封皮插两遍：先比对目标 PDF 的**最后一页**是否已与源 PDF 末页一致
（尺寸 + 页内图像哈希），一致则跳过。
"""
import os
import sys
import argparse
import hashlib

import pymupdf


# ---------- 工具函数 ----------

def _page_fingerprint(doc, pi):
    """给一页算指纹：页面尺寸 + 页内图片的 MD5。用于判断「是否同一页」。"""
    if pi < 0 or pi >= doc.page_count:
        return None
    pg = doc[pi]
    rect = pg.rect
    h = hashlib.md5()
    h.update(f'{rect.width:.1f}x{rect.height:.1f}'.encode())
    # 把页内所有图片的原始字节喂进哈希
    for img in pg.get_images(full=True):
        xref = img[0]
        try:
            info = doc.extract_image(xref)
            h.update(info['image'])
        except Exception:
            pass
    # 无图片的页（纯文本），退化为用页面绘制内容哈希
    if not pg.get_images():
        pix = pg.get_pixmap(dpi=40)
        h.update(pix.samples)
    return h.hexdigest()


def _natural_last_pdf(sub_dir):
    """在子文件夹里找出「编号最大」的 PDF（001.pdf < 002.pdf < ...）。"""
    pdfs = [f for f in os.listdir(sub_dir)
            if f.lower().endswith('.pdf') and not f.lower().endswith('.bak.pdf')]
    if not pdfs:
        return None

    def key(fn):
        stem = os.path.splitext(fn)[0]
        # 纯数字按数值排（001 -> 1）；非纯数字排到最后并按其字符串
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return max(pdfs, key=key)


# ---------- 单份处理 ----------

def process_one(src_pdf, sub_dir, dry_run=False, backup=True, log=print):
    """把 src_pdf 的最后一页追加到 sub_dir 里编号最大的 PDF 末尾。

    返回 'done' / 'skip_same' / 'skip_nofile' / 'error:...'
    """
    name = os.path.basename(sub_dir)
    if not os.path.isfile(src_pdf):
        log(f'  ! 源 PDF 不存在: {src_pdf}')
        return 'skip_nofile'

    last_fn = _natural_last_pdf(sub_dir)
    if last_fn is None:
        log(f'  ! 结果文件夹里没有 PDF: {sub_dir}')
        return 'skip_nofile'
    last_path = os.path.join(sub_dir, last_fn)

    src = pymupdf.open(src_pdf)
    try:
        src_last = src.page_count - 1
        fp_src = _page_fingerprint(src, src_last)

        tgt = pymupdf.open(last_path)
        try:
            # --- 幂等检查：目标末页是否已是封皮 ---
            fp_tgt_last = _page_fingerprint(tgt, tgt.page_count - 1)
            if fp_src is not None and fp_tgt_last == fp_src:
                log(f'  = 已是封皮，跳过: {name}/{last_fn}')
                return 'skip_same'

            tgt_pages_before = tgt.page_count
            if dry_run:
                log(f'  [dry-run] 将把 {name}.pdf 末页(第{src_last+1}页)'
                    f' 追加到 {name}/{last_fn} (现 {tgt_pages_before} 页)')
                return 'done'

            # --- 备份 ---
            if backup:
                bak = last_path[:-4] + '.bak.pdf'
                if not os.path.exists(bak):
                    tgt.save(bak)
                    log(f'  · 已备份原文件 -> {os.path.basename(bak)}')

            # --- 追加封皮页 ---
            tgt.insert_pdf(src, from_page=src_last, to_page=src_last)
            tmp = last_path + '.tmp'
            tgt.save(tmp, garbage=3, deflate=True)
        finally:
            tgt.close()
    finally:
        src.close()

    os.replace(tmp, last_path)
    log(f'  + 已追加封皮 -> {name}/{last_fn}'
        f' ({tgt_pages_before} -> {tgt_pages_before + 1} 页)')
    return 'done'


def run(src_dir, out_dir, dry_run=False, backup=True, log=print):
    """遍历 out_dir 下每个子文件夹，与 src_dir 同名 PDF 配对处理。"""
    if not os.path.isdir(src_dir):
        log(f'!! 源 PDF 目录不存在: {src_dir}')
        return
    if not os.path.isdir(out_dir):
        log(f'!! 拆分结果目录不存在: {out_dir}')
        return

    subs = sorted(d for d in os.listdir(out_dir)
                  if os.path.isdir(os.path.join(out_dir, d)))
    if not subs:
        log('!! 结果目录下没有子文件夹')
        return

    log(f'发现 {len(subs)} 个结果文件夹，开始配对…\n')
    stat = {'done': 0, 'skip_same': 0, 'skip_nofile': 0, 'error': 0}
    for sub in subs:
        sub_dir = os.path.join(out_dir, sub)
        src_pdf = os.path.join(src_dir, sub + '.pdf')
        log(f'===== {sub} =====')
        if not os.path.isfile(src_pdf):
            # 兜底：源目录里可能有前缀/空格差异，做一次模糊匹配
            cand = _fuzzy_find(src_dir, sub)
            if cand:
                src_pdf = cand
                log(f'  (按前缀匹配到: {os.path.basename(cand)})')
            else:
                log(f'  ! 源目录里找不到同名 PDF（找过: {sub}.pdf）')
                stat['skip_nofile'] += 1
                continue
        try:
            r = process_one(src_pdf, sub_dir, dry_run=dry_run,
                            backup=backup, log=log)
        except Exception as e:
            log(f'  !! 处理出错: {type(e).__name__}: {e}')
            stat['error'] += 1
            continue
        stat['done' if r == 'done' else r] = stat.get(
            r if r in stat else 'error', 0) + 1
    log(f'\n完成：追加 {stat["done"]} 个，跳过(已是封皮) {stat["skip_same"]} 个，'
        f'未配对 {stat["skip_nofile"]} 个，出错 {stat["error"]} 个')
    if dry_run:
        log('（--dry-run 模式，未实际写文件）')


def _fuzzy_find(src_dir, stem):
    """源文件名与文件夹名可能只差少量字符（空格/半角全角），做宽松匹配。"""
    import re
    norm = lambda s: re.sub(r'[\s\-_·•]', '', s).lower()
    target = norm(stem)
    for f in os.listdir(src_dir):
        if not f.lower().endswith('.pdf'):
            continue
        if norm(os.path.splitext(f)[0]) == target:
            return os.path.join(src_dir, f)
    # 再退一步：互为前缀
    for f in os.listdir(src_dir):
        if not f.lower().endswith('.pdf'):
            continue
        base = norm(os.path.splitext(f)[0])
        if base.startswith(target) or target.startswith(base):
            return os.path.join(src_dir, f)
    return None


def main():
    ap = argparse.ArgumentParser(description='把源 PDF 末页(封皮)追加到拆分结果的最后一个 PDF')
    ap.add_argument('--src', required=True, help='源 PDF 所在目录')
    ap.add_argument('--out', required=True, help='拆分结果目录（内含各卷宗子文件夹）')
    ap.add_argument('--dry-run', action='store_true', help='只预览，不写文件')
    ap.add_argument('--no-backup', action='store_true', help='不备份原文件（默认备份）')
    a = ap.parse_args()
    run(a.src, a.out, dry_run=a.dry_run, backup=not a.no_backup)


if __name__ == '__main__':
    main()
