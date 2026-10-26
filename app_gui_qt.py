# -*- coding: utf-8 -*-
"""
PDF 智能拆分 — Exe 图形界面入口（PySide6 版）
============================================
为什么用 PySide6: 打包用的隔离 Python 没有 tkinter(未编译 _tkinter.pyd),
故改用纯 pip 可装的 Qt 绑定。功能与 tkinter 版(app_gui.py)完全一致。

给非技术用户: 选一个放 PDF 的文件夹 -> 选输出文件夹 -> 点"开始拆分"。
"""
import os
import sys
import traceback

# ---- 让打包后的 exe 能找到同目录的 scripts / 模型 ----
if getattr(sys, 'frozen', False):
    BASE = os.path.dirname(sys.executable)
    BUNDLE = sys._MEIPASS
    sys.path.insert(0, os.path.join(BUNDLE, 'scripts'))
    sys.path.insert(0, os.path.join(BUNDLE, 'runtime'))
    _px = os.path.join(BUNDLE, 'runtime', 'paddlex')
    if os.path.isdir(_px):
        os.environ['PADDLE_PDX_CACHE_HOME'] = _px
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLE = BASE
    sys.path.insert(0, BASE)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
    QPlainTextEdit, QProgressBar)


class Worker(QThread):
    """后台跑拆分, 通过信号回传日志(避免阻塞界面)。"""
    log_sig = Signal(str)
    done_sig = Signal()

    def __init__(self, in_dir, out_dir, with_cover, fast):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.with_cover = with_cover
        self.fast = fast

    def run(self):
        try:
            import core_split
            core_split.run(self.in_dir, self.out_dir,
                           with_cover=self.with_cover, fast=self.fast,
                           log=lambda s: self.log_sig.emit(str(s)))
            self.log_sig.emit('\n全部完成。')
        except Exception as e:
            self.log_sig.emit('出错: ' + repr(e))
            self.log_sig.emit(traceback.format_exc())
        finally:
            self.done_sig.emit()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PDF 智能拆分 — 档案卷宗按卷内目录拆分')
        self.resize(820, 600)
        self.worker = None

        root = QVBoxLayout(self)
        grid = QGridLayout()
        root.addLayout(grid)

        # 输入目录
        grid.addWidget(QLabel('PDF 所在文件夹:'), 0, 0)
        self.in_edit = QLineEdit()
        grid.addWidget(self.in_edit, 0, 1)
        b1 = QPushButton('选择…')
        b1.clicked.connect(self.pick_in)
        grid.addWidget(b1, 0, 2)

        # 输出目录
        grid.addWidget(QLabel('输出文件夹:'), 1, 0)
        self.out_edit = QLineEdit()
        grid.addWidget(self.out_edit, 1, 1)
        b2 = QPushButton('选择…')
        b2.clicked.connect(self.pick_out)
        grid.addWidget(b2, 1, 2)

        # 选项
        opt = QHBoxLayout()
        self.cover_cb = QCheckBox('把封面和目录并入第一个 PDF')
        self.cover_cb.setChecked(True)
        opt.addWidget(self.cover_cb)
        self.fast_cb = QCheckBox(
            '快速模式（抽样12页，快约14倍；取消则逐条OCR校验，慢但最准）')
        self.fast_cb.setChecked(True)
        opt.addWidget(self.fast_cb)
        opt.addStretch()
        root.addLayout(opt)

        # 开始按钮
        self.btn = QPushButton('开始拆分')
        self.btn.setMinimumHeight(38)
        self.btn.setStyleSheet(
            'QPushButton{background:#2b7de9;color:white;font-size:14px;'
            'border-radius:6px;} QPushButton:disabled{background:#9bb8e0;}')
        self.btn.clicked.connect(self.start)
        root.addWidget(self.btn)

        # 进度条
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)      # 不确定进度(忙碌态)
        self.bar.setVisible(False)
        root.addWidget(self.bar)

        # 日志
        root.addWidget(QLabel('运行日志:'))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family:Consolas,monospace;font-size:12px;')
        root.addWidget(self.log, 1)

        self.print('请选择 PDF 所在文件夹与输出文件夹，然后点「开始拆分」。')

    def print(self, s):
        self.log.appendPlainText(s)

    def pick_in(self):
        d = QFileDialog.getExistingDirectory(self, '选择放 PDF 的文件夹')
        if d:
            self.in_edit.setText(d)
            if not self.out_edit.text():
                self.out_edit.setText(os.path.join(d, '拆分结果'))

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, '选择输出文件夹')
        if d:
            self.out_edit.setText(d)

    def start(self):
        ind, outd = self.in_edit.text().strip(), self.out_edit.text().strip()
        if not ind or not outd:
            QMessageBox.warning(self, '提示', '请先选择输入和输出文件夹')
            return
        self.btn.setEnabled(False)
        self.bar.setVisible(True)
        self.worker = Worker(ind, outd, self.cover_cb.isChecked(),
                             self.fast_cb.isChecked())
        self.worker.log_sig.connect(self.print)
        self.worker.done_sig.connect(self.finish)
        self.worker.start()

    def finish(self):
        self.btn.setEnabled(True)
        self.bar.setVisible(False)


def headless_main():
    """隐藏的无界面自检模式: 用于打包后验证 OCR 链路可用。

    用法(仅测试用, 普通用户不会碰到):
        PDFSplit.exe --selftest <输入目录> <输出目录> [fast|accurate]

    之所以要这个模式: exe 是 --noconsole 的, 无法直接用 print 观察;
    有了它就能用脚本驱动 exe 跑真实数据, 确认模型能加载、能出 PDF。
    """
    argv = sys.argv[1:]
    i = argv.index('--selftest')
    in_dir = argv[i + 1]
    out_dir = argv[i + 2]
    fast = not (len(argv) > i + 3 and argv[i + 3] == 'accurate')
    import core_split
    def log(msg):
        try:
            with open(os.path.join(out_dir, '_selftest.log'), 'a', encoding='utf-8') as f:
                f.write(str(msg) + '\n')
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    try:
        core_split.run(in_dir, out_dir, with_cover=True, fast=fast, log=log)
        log('SELFTEST_OK')
    except Exception:
        log('SELFTEST_FAIL\n' + traceback.format_exc())


def main():
    if '--selftest' in sys.argv:
        headless_main()
        return
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
