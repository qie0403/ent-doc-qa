"""
生成依赖锁定文件 requirements.lock.txt
=====================================

作用：
    离线解析 requirements.txt 中直接依赖的传递闭包，
    结合当前环境已安装的版本，生成精确版本锁定文件，
    保证他人（以及 CI / Docker）能复现同一套依赖组合。

运行：
    python scripts/gen_lock.py

说明：
    - 不访问网络、不安装/卸载任何包，只读取已安装包的元数据；
    - 自动跳过 extra 可选依赖与当前环境不匹配的条件依赖；
    - 输出为 UTF-8 的 requirements.lock.txt。
"""

import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQ_FILE = PROJECT_ROOT / "requirements.txt"
OUT_FILE = PROJECT_ROOT / "requirements.lock.txt"


def norm(name: str) -> str:
    """规范化包名（PEP 503）"""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def parse_direct_requirements():
    """从 requirements.txt 解析直接依赖"""
    direct = []
    for line in REQ_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[;]", s)[0].strip()
        if name:
            direct.append((name, s))
    return direct


def build_installed_index():
    """建立 规范化包名 -> (原始名, 版本) 的映射"""
    installed = {}
    for dist in metadata.distributions():
        try:
            n = dist.metadata["Name"]
            v = dist.version
        except Exception:
            continue
        if not n:
            continue
        installed[norm(n)] = (n, v)
    return installed


def resolve_closure(direct, installed):
    """广度优先解析传递闭包"""
    closure = {}
    missing = []
    queue = [norm(n) for n, _ in direct]
    seen = set()

    while queue:
        key = queue.pop(0)
        if key in seen:
            continue
        seen.add(key)

        if key not in installed:
            missing.append(key)
            continue

        name, ver = installed[key]
        closure[key] = (name, ver)

        try:
            reqs = metadata.distribution(name).requires or []
        except Exception:
            reqs = []

        for r in reqs:
            # 跳过 extra 可选依赖
            if "extra ==" in r or "extra==" in r:
                continue
            try:
                req_obj = Requirement(r)
            except Exception as exc:
                print(f"  [warn] 无法解析依赖串 {r!r}: {exc}")
                continue
            # 跳过当前环境不匹配的条件依赖
            if req_obj.marker is not None and not req_obj.marker.evaluate():
                continue
            queue.append(norm(req_obj.name))

    return closure, missing


def main():
    direct = parse_direct_requirements()
    installed = build_installed_index()
    closure, missing = resolve_closure(direct, installed)

    lines = [
        "# ============================================",
        "# 依赖锁定文件（传递闭包，精确版本）",
        "# ============================================",
        "# 生成方式：python scripts/gen_lock.py（离线解析依赖树）",
        f"# 验证环境：Python {sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro} / {sys.platform}",
        f"# 直接依赖 {len(direct)} 个，传递闭包共 {len(closure)} 个包",
        "#",
        "# 用法：",
        "#   pip install -r requirements.lock.txt   # 完全复现验证环境（CI / 生产）",
        "#   pip install -r requirements.txt        # 允许较新版本（日常开发）",
        "#",
        "# 重新生成：升级依赖后重跑 python scripts/gen_lock.py",
        "# ============================================",
        "",
        "# ---- 直接依赖（requirements.txt 中声明）----",
    ]

    for name, spec in direct:
        key = norm(name)
        if key in closure:
            n, v = closure[key]
            lines.append(f"{n}=={v}")
        else:
            lines.append(f"# MISSING {spec}")

    lines += ["", "# ---- 传递依赖（自动解析）----"]
    direct_keys = {norm(n) for n, _ in direct}
    for key in sorted(closure):
        if key in direct_keys:
            continue
        n, v = closure[key]
        lines.append(f"{n}=={v}")
    lines.append("")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"直接依赖: {len(direct)}")
    print(f"闭包总数: {len(closure)}")
    print(f"未安装（应为 0）: {missing}")
    print(f"已写入: {OUT_FILE}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
