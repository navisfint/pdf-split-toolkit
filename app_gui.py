# -*- coding: utf-8 -*-
"""
PDF 智能拆分 — Exe 图形界面入口
================================
给非技术用户使用: 选一个放 PDF 的文件夹 -> 选输出文件夹 -> 点"开始拆分"。
程序自动: 探测每份 PDF 的卷内目录页 -> 解析目录 -> 推算边界 -> 切分 -> 出报告

打包: pyinstaller --onefile --noconsole --name PDF智能拆分 app_gui.py
"""
import os
import sys
import json
import time
import threading
import traceback

# ---- 让打包后的 exe 能找到同目录的 scripts / 模型 ----
if getattr(sys, 'frozen', False):
    BASE = os.path.dirname(sys.executable)
    BUNDLE = sys._MEIPASS
    sys.path.insert(0, os.path.join(BUNDLE, 'scripts'))
    sys.path.insert(0, os.path.join(BUNDLE, 'runtime'))
    # 把 PaddleX 的模型缓存目录指向 exe 内自带的 runtime/paddlex
    _px = os.path.join(BUNDLE, 'runtime', 'paddlex')
    if os.path.isdir(_px):
        os.environ['PADDLE_PDX_CACHE_HOME'] = _px
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLE = BASE
    sys.path.insert(0, BASE)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


class App:
    def __init__(self, root):
        self.root = root
        root.title('PDF 智能拆分 — 档案卷宗按卷内目录拆分')
        root.geometry('760x560')

        frm = tk.Frame(root, padx=14, pady=12)
        frm.pack(fill='both', expand=True)

        # 输入目录
        tk.Label(frm, text='PDF 所在文件夹:', anchor='w').grid(
            row=0, column=0, sticky='w')
        self.in_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.in_var, width=62).grid(
            row=0, column=1, sticky='we', padx=4)
        tk.Button(frm, text='选择…', command=self.pick_in).grid(row=0, column=2)

        # 输出目录
        tk.Label(frm, text='输出文件夹:', anchor='w').grid(
            row=1, column=0, sticky='w', pady=(6, 0))
        self.out_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.out_var, width=62).grid(
            row=1, column=1, sticky='we', padx=4, pady=(6, 0))
        tk.Button(frm, text='选择…', command=self.pick_out).grid(
            row=1, column=2, pady=(6, 0))

        # 选项
        opt = tk.Frame(frm)
        opt.grid(row=2, column=0, columnspan=3, sticky='w', pady=(10, 4))
        self.cover_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text='把封面和目录并入第一个 PDF',
                       variable=self.cover_var).pack(side='left')
        self.fast_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text='快速模式（抽样12页，快约14倍；取消则逐条OCR校验，慢但最准）',
                       variable=self.fast_var).pack(side='left', padx=(16, 0))

        # 开始按钮
        self.btn = tk.Button(frm, text='开始拆分', height=2,
                             bg='#2b7de9', fg='white',
                             command=self.start)
        self.btn.grid(row=3, column=0, columnspan=3, sticky='we', pady=(4, 8))

        # 日志
        tk.Label(frm, text='运行日志:', anchor='w').grid(
            row=4, column=0, columnspan=3, sticky='w')
        self.log = scrolledtext.ScrolledText(frm, height=20, font=('Consolas', 9))
        self.log.grid(row=5, column=0, columnspan=3, sticky='nsew')
        frm.rowconfigure(5, weight=1)
        frm.columnconfigure(1, weight=1)

        self.print('请选择 PDF 所在文件夹与输出文件夹，然后点「开始拆分」。')

    def print(self, s):
        self.log.insert('end', s + '\n')
        self.log.see('end')
        self.root.update_idletasks()

    def pick_in(self):
        d = filedialog.askdirectory(title='选择放 PDF 的文件夹')
        if d:
            self.in_var.set(d)
            if not self.out_var.get():
                self.out_var.set(os.path.join(d, '拆分结果'))

    def pick_out(self):
        d = filedialog.askdirectory(title='选择输出文件夹')
        if d:
            self.out_var.set(d)

    def start(self):
        if not self.in_var.get() or not self.out_var.get():
            messagebox.showwarning('提示', '请先选择输入和输出文件夹')
            return
        self.btn.config(state='disabled')
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def run(self):
        try:
            import core_split
            core_split.run(
                self.in_var.get(), self.out_var.get(),
                with_cover=self.cover_var.get(),
                fast=self.fast_var.get(),
                log=self.print)
            self.print('\n全部完成。')
        except Exception as e:
            self.print('出错: ' + repr(e))
            self.print(traceback.format_exc())
        finally:
            self.btn.config(state='normal')


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
