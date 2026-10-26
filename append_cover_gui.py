# -*- coding: utf-8 -*-
"""
封皮追加工具 — 图形界面入口（PySide6 版）
==========================================
用途: PDF 拆分后, 把源 PDF 最后一页(牛皮纸「卷内备考表」)补进
      各卷宗"编号最大"的那个拆分 PDF 的末尾。

给非技术用户: 选「源 PDF 文件夹」+「拆分结果文件夹」-> 点一下即可。
本程序不依赖 OCR / Paddle, 启动很快, 体积也小。
"""
import os
import sys
import traceback

if getattr(sys, 'frozen', False):
    BASE = os.path.dirname(sys.executable)
    BUNDLE = sys._MEIPASS
    sys.path.insert(0, BUNDLE)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
    QPlainTextEdit, QProgressBar)


class Worker(QThread):
    log_sig = Signal(str)
    done_sig = Signal()

    def __init__(self, src_dir, out_dir, dry_run, backup):
        super().__init__()
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.backup = backup

    def run(self):
        try:
            import append_cover
            append_cover.run(self.src_dir, self.out_dir,
                             dry_run=self.dry_run, backup=self.backup,
                             log=lambda s: self.log_sig.emit(str(s)))
            self.log_sig.emit('\n处理结束。')
        except Exception as e:
            self.log_sig.emit('出错: ' + repr(e))
            self.log_sig.emit(traceback.format_exc())
        finally:
            self.done_sig.emit()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('封皮追加工具 — 把源PDF末页补进最后一个拆分PDF')
        self.resize(820, 580)
        self.worker = None

        root = QVBoxLayout(self)
        grid = QGridLayout()
        root.addLayout(grid)

        # 源 PDF 目录
        grid.addWidget(QLabel('源 PDF 文件夹:'), 0, 0)
        self.src_edit = QLineEdit()
        grid.addWidget(self.src_edit, 0, 1)
        b1 = QPushButton('选择…')
        b1.clicked.connect(self.pick_src)
        grid.addWidget(b1, 0, 2)

        # 拆分结果目录
        grid.addWidget(QLabel('拆分结果文件夹:'), 1, 0)
        self.out_edit = QLineEdit()
        grid.addWidget(self.out_edit, 1, 1)
        b2 = QPushButton('选择…')
        b2.clicked.connect(self.pick_out)
        grid.addWidget(b2, 1, 2)

        # 选项
        opt = QHBoxLayout()
        self.bak_cb = QCheckBox('改写前备份原文件（.bak.pdf）')
        self.bak_cb.setChecked(True)
        opt.addWidget(self.bak_cb)
        self.dry_cb = QCheckBox('仅预览不写文件（试运行）')
        opt.addWidget(self.dry_cb)
        opt.addStretch()
        root.addLayout(opt)

        tip = QLabel('说明：会把「源 PDF 的最后一页」追加到每个卷宗结果文件夹里'
                     '编号最大的那个 PDF 末尾。重复运行不会重复插入。')
        tip.setWordWrap(True)
        tip.setStyleSheet('color:#666;font-size:12px;')
        root.addWidget(tip)

        # 按钮
        self.btn = QPushButton('开始追加封皮')
        self.btn.setMinimumHeight(38)
        self.btn.setStyleSheet(
            'QPushButton{background:#2b7de9;color:white;font-size:14px;'
            'border-radius:6px;} QPushButton:disabled{background:#9bb8e0;}')
        self.btn.clicked.connect(self.start)
        root.addWidget(self.btn)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        root.addWidget(self.bar)

        root.addWidget(QLabel('运行日志:'))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet('font-family:Consolas,monospace;font-size:12px;')
        root.addWidget(self.log, 1)

        self.print('请选择「源 PDF 文件夹」和「拆分结果文件夹」，然后点「开始追加封皮」。')

    def print(self, s):
        self.log.appendPlainText(s)

    def pick_src(self):
        d = QFileDialog.getExistingDirectory(self, '选择放源 PDF 的文件夹')
        if d:
            self.src_edit.setText(d)

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, '选择拆分结果文件夹')
        if d:
            self.out_edit.setText(d)

    def start(self):
        sd, od = self.src_edit.text().strip(), self.out_edit.text().strip()
        if not sd or not od:
            QMessageBox.warning(self, '提示', '请先选择源 PDF 文件夹和拆分结果文件夹')
            return
        if not os.path.isdir(sd) or not os.path.isdir(od):
            QMessageBox.warning(self, '提示', '目录不存在，请重新选择')
            return
        self.btn.setEnabled(False)
        self.bar.setVisible(True)
        self.worker = Worker(sd, od, self.dry_cb.isChecked(),
                             self.bak_cb.isChecked())
        self.worker.log_sig.connect(self.print)
        self.worker.done_sig.connect(self.finish)
        self.worker.start()

    def finish(self):
        self.btn.setEnabled(True)
        self.bar.setVisible(False)


def headless_main():
    """隐藏的无界面自检模式（仅测试用）: 驱动打包后的 exe 跑真实数据。
    用法: 封皮追加工具.exe --selftest <源目录> <结果目录> <日志文件> [--dry-run]
    """
    argv = sys.argv[1:]
    i = argv.index('--selftest')
    src_dir, out_dir, log_path = argv[i + 1], argv[i + 2], argv[i + 3]
    dry = '--dry-run' in argv
    import append_cover
    lines = []
    def log(m):
        lines.append(str(m))
    try:
        append_cover.run(src_dir, out_dir, dry_run=dry, backup=False, log=log)
        lines.append('SELFTEST_OK')
    except Exception:
        lines.append('SELFTEST_FAIL\n' + traceback.format_exc())
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


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
