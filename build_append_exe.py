# -*- coding: utf-8 -*-
"""
打包「封皮追加工具」为独立单文件 exe
=====================================
用法: python build_append_exe.py

产出: dist/封皮追加工具.exe

与主程序不同，本工具**不需要 PaddleOCR / 模型**，只用到 pymupdf + PySide6，
所以 exe 体积会小很多（约 60~90MB）。打包前请确保已装 pymupdf、PySide6。
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENTRY = 'append_cover_gui.py'
NAME = '封皮追加工具'


def main():
    os.chdir(HERE)

    cmd = [PY, '-m', 'PyInstaller',
           '--noconfirm',
           '--onefile', '--noconsole',
           '--name', NAME,
           # 只保留界面用到的 Qt 模块，排除其余大模块(与主程序同一套策略)
           '--exclude-module', 'PySide6.QtQml',
           '--exclude-module', 'PySide6.QtQuick',
           '--exclude-module', 'PySide6.QtQuick3D',
           '--exclude-module', 'PySide6.QtQuickWidgets',
           '--exclude-module', 'PySide6.QtWebEngineCore',
           '--exclude-module', 'PySide6.QtWebEngineWidgets',
           '--exclude-module', 'PySide6.QtWebEngineQuick',
           '--exclude-module', 'PySide6.QtWebChannel',
           '--exclude-module', 'PySide6.QtWebSockets',
           '--exclude-module', 'PySide6.QtMultimedia',
           '--exclude-module', 'PySide6.QtMultimediaWidgets',
           '--exclude-module', 'PySide6.QtCharts',
           '--exclude-module', 'PySide6.QtDataVisualization',
           '--exclude-module', 'PySide6.QtGraphs',
           '--exclude-module', 'PySide6.Qt3DCore',
           '--exclude-module', 'PySide6.Qt3DRender',
           '--exclude-module', 'PySide6.Qt3DInput',
           '--exclude-module', 'PySide6.Qt3DLogic',
           '--exclude-module', 'PySide6.Qt3DAnimation',
           '--exclude-module', 'PySide6.Qt3DExtras',
           '--exclude-module', 'PySide6.QtSql',
           '--exclude-module', 'PySide6.QtTest',
           '--exclude-module', 'PySide6.QtDesigner',
           '--exclude-module', 'PySide6.QtHelp',
           '--exclude-module', 'PySide6.QtUiTools',
           '--exclude-module', 'PySide6.QtSvg',
           '--exclude-module', 'PySide6.QtSvgWidgets',
           '--exclude-module', 'PySide6.QtSerialPort',
           '--exclude-module', 'PySide6.QtPositioning',
           '--exclude-module', 'PySide6.QtBluetooth',
           '--exclude-module', 'PySide6.QtNfc',
           '--exclude-module', 'PySide6.QtPdf',
           '--exclude-module', 'PySide6.QtPdfWidgets',
           '--exclude-module', 'PySide6.QtScxml',
           '--exclude-module', 'PySide6.QtStateMachine',
           '--exclude-module', 'PySide6.QtVirtualKeyboard',
           '--exclude-module', 'PySide6.QtRemoteObjects',
           '--exclude-module', 'PySide6.QtSpatialAudio',
           '--exclude-module', 'PySide6.QtTextToSpeech',
           '--exclude-module', 'PySide6.QtHttpServer',
           '--exclude-module', 'PySide6.QtNetworkAuth',
           '--exclude-module', 'PySide6.QtConcurrent',
           '--exclude-module', 'PySide6.scripts',
           '--exclude-module', 'PySide6.assistant',
           '--exclude-module', 'PySide6.designer',
           '--exclude-module', 'PySide6.linguist',
           '--exclude-module', 'shiboken6_generator',
           # 本工具完全不需要 OCR 栈，主动排除以免被误收集
           '--exclude-module', 'paddle',
           '--exclude-module', 'paddleocr',
           '--exclude-module', 'paddlex',
           '--exclude-module', 'scipy',
           '--exclude-module', 'skimage',
           '--exclude-module', 'cv2',
           '--exclude-module', 'numpy',
           '--hidden-import', 'PySide6.QtCore',
           '--hidden-import', 'PySide6.QtGui',
           '--hidden-import', 'PySide6.QtWidgets',
           ENTRY]

    print('执行 PyInstaller…')
    r = subprocess.run(cmd)
    print('PyInstaller 退出码:', r.returncode)
    exe = os.path.join(HERE, 'dist', NAME + '.exe')
    if os.path.exists(exe):
        print(f'✅ 打包完成: {exe} ({os.path.getsize(exe)/1024/1024:.0f} MB)')
    else:
        print('❌ 未找到 exe, 请检查上面的报错')


if __name__ == '__main__':
    main()
