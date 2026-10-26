# -*- coding: utf-8 -*-
"""
PyInstaller 打包脚本 — 生成单文件 exe
====================================
用法: python build_exe.py

产出: dist/PDF智能拆分.exe

要点:
  1. --onefile 单文件; --noconsole 不弹黑窗
  2. 把 scripts/ 下所有模块 + 运行期模型一起打进 exe
  3. PaddleOCR 模型(~/.paddlex, 177MB)复制到 runtime/paddlex, 运行时优先使用
  4. 用 --collect-all 收全 paddle/paddleocr/pymupdf/scipy 等隐式依赖
  5. GUI 用 PySide6(而非 tkinter): 本机隔离 Python 未编译 _tkinter.pyd,
     打包后 exe 无 tkinter -> 双击闪退 ModuleNotFoundError: tkinter
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# GUI 入口: PySide6 版(不依赖 tkinter)
ENTRY = 'app_gui_qt.py'


def main():
    os.chdir(HERE)

    # 1) 准备 runtime 目录(把 PaddleOCR 模型拷进来)
    runtime = os.path.join(HERE, 'runtime')
    target_models = os.path.join(runtime, 'paddlex')
    src_models = os.path.join(os.path.expanduser('~'), '.paddlex')
    if os.path.isdir(src_models) and not os.path.isdir(target_models):
        print('复制 PaddleOCR 模型…')
        shutil.copytree(src_models, target_models)

    # 2) 组装 pyinstaller 命令
    sep = ';' if os.name == 'nt' else ':'
    add_data = [
        # (源, exe 内目标路径)
        (os.path.join(HERE, 'scripts'), 'scripts'),
        (os.path.join(HERE, 'core_split.py'), '.'),
    ]
    if os.path.isdir(target_models):
        add_data.append((target_models, 'runtime/paddlex'))

    cmd = [PY, '-m', 'PyInstaller',
           '--noconfirm',
           '--onefile', '--noconsole',
           '--name', 'PDF智能拆分',
           # --- 压缩体积: 显式排除用不到的 Qt 大模块 ---
           # GUI 只用到 QtCore/QtGui/QtWidgets; 不排除的话 PySide6 会把
           # Qt3D/Qml/Quick/WebEngine/Multimedia 整套塞进来(多出 ~300MB)
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
           '--exclude-module', 'PySide6.QtOpenGL',
           '--exclude-module', 'PySide6.QtOpenGLWidgets',
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
           # PySide6 附带的工具/绑定(非运行时必需), 也一并排除
           '--exclude-module', 'PySide6.scripts',
           '--exclude-module', 'PySide6.assistant',
           '--exclude-module', 'PySide6.designer',
           '--exclude-module', 'PySide6.linguist',
           '--exclude-module', 'shiboken6_generator',
           '--collect-all', 'paddle',
           '--collect-all', 'paddleocr',
           '--collect-all', 'paddlex',
           '--collect-all', 'pymupdf',
           '--collect-all', 'scipy',
           '--collect-all', 'skimage',
           '--collect-submodules', 'toc_parser',
           ]
    for s, d in add_data:
        cmd += ['--add-data', f'{s}{sep}{d}']
    # 隐藏导入(动态 import 的模块)
    for m in ['page_map_engine', 'toc_parser', 'offset_fitter',
              'PIL', 'PIL.Image', 'numpy', 'cv2']:
        cmd += ['--hidden-import', m]
    # ★关键: PaddleX 的依赖自检走 importlib.metadata(get_dep_version),
    #   不是 import。PyInstaller 默认只带"被 import 的包"的 dist-info,
    #   其余缺失 -> require_extra 判定不可用 -> 启动即报
    #   DependencyError: `OCR` requires additional dependencies。
    #
    #   PaddleOCR 走的是 paddlex[ocr] 或 alt=paddlex[ocr-core];
    #   ocr-core 只需下面 6 个, 且它们在环境里都有 dist-info。
    #   (paddlex[ocr] 的另外 15 个在环境里本就没装, 故不能 --copy-metadata,
    #    否则打包直接报错; 好在 alt 子集满足即可通过校验。)
    #   清单来源: paddlex.utils.deps.EXTRAS['ocr-core']
    for m in ['imagesize', 'opencv-contrib-python', 'pyclipper',
              'pypdfium2', 'python-bidi', 'shapely']:
        cmd += ['--copy-metadata', m]
    # 这些小包无静态 import 时可能不进包体, 显式声明入口模块名
    for m in ['imagesize', 'pyclipper', 'pypdfium2', 'bidi']:
        cmd += ['--hidden-import', m]
    # PySide6 仅显式声明界面真正用到的三个模块(替代原来的 --collect-all PySide6)
    for m in ['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets']:
        cmd += ['--hidden-import', m]
    cmd += [ENTRY]

    print('执行:', ' '.join(cmd[:8]), '…')
    r = subprocess.run(cmd)
    print('PyInstaller 退出码:', r.returncode)
    exe = os.path.join(HERE, 'dist', 'PDF智能拆分.exe')
    if os.path.exists(exe):
        print(f'✅ 打包完成: {exe} ({os.path.getsize(exe)/1024/1024:.0f} MB)')
    else:
        print('❌ 未找到 exe, 请检查上面的报错')


if __name__ == '__main__':
    main()
