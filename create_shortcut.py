"""
create_shortcut.py - 在桌面创建 NeRF SR Studio 快捷方式

运行方式: python create_shortcut.py
效果: 在桌面生成 "NeRF SR Studio.lnk"，双击即可启动后端服务器并打开浏览器
"""
import os
import subprocess
import sys


def create_desktop_shortcut():
    # 项目根目录和启动脚本路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    bat_path = os.path.join(project_dir, "start_studio.bat")

    if not os.path.exists(bat_path):
        print(f"[错误] 启动脚本不存在: {bat_path}")
        return False

    # 使用 VBScript 创建快捷方式（无 PowerShell 执行策略限制）
    import tempfile
    import ctypes

    # 通过 Windows Shell API 获取桌面路径（最可靠）
    # CSIDL_DESKTOPDIRECTORY = 0x0010
    desktop = None
    buf = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(0, 0x0010, 0, 0, buf) == 0:
        if os.path.isdir(buf.value):
            desktop = buf.value

    # 后备方案
    if not desktop:
        for candidate in [
            os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
            os.path.join(os.environ.get("USERPROFILE", ""), "桌面"),
        ]:
            if os.path.isdir(candidate):
                desktop = candidate
                break

    if not desktop:
        print("[错误] 无法找到桌面路径")
        return False

    shortcut_path = os.path.join(desktop, "NeRF SR Studio.lnk")

    # 写入临时 VBScript 文件并执行
    vbs_content = (
        'Set ws = CreateObject("WScript.Shell")\n'
        f'Set sc = ws.CreateShortcut("{shortcut_path}")\n'
        f'sc.TargetPath = "{bat_path}"\n'
        f'sc.WorkingDirectory = "{project_dir}"\n'
        'sc.Description = "NeRF SR Studio"\n'
        'sc.WindowStyle = 1\n'
        'sc.Save\n'
    )

    vbs_path = os.path.join(tempfile.gettempdir(), "create_nerf_shortcut.vbs")
    try:
        with open(vbs_path, "w", encoding="gbk") as f:
            f.write(vbs_content)

        result = subprocess.run(
            ["cscript", "//Nologo", vbs_path],
            capture_output=True, timeout=10
        )

        # 清理临时文件
        os.remove(vbs_path)

        if result.returncode != 0:
            print(f"[错误] VBScript 执行失败 (返回码 {result.returncode})")
            return False

        if not os.path.exists(shortcut_path):
            print("[错误] 快捷方式文件未生成")
            return False

    except Exception as e:
        print(f"[错误] 创建快捷方式失败: {e}")
        if os.path.exists(vbs_path):
            os.remove(vbs_path)
        return False

    print(f"桌面快捷方式已创建: {shortcut_path}")
    print(f"目标: {bat_path}")
    print(f"\n双击桌面上的 'NeRF SR Studio' 即可一键启动！")
    return True


if __name__ == "__main__":
    create_desktop_shortcut()
