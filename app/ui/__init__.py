"""Streamlit UI 包。

入口：`app/ui/streamlit_app.py`，启动方式：

    streamlit run app/ui/streamlit_app.py

**为什么放在 `app/ui/` 而不是仓库根**：Streamlit 的惯例是放根目录
（`app.py` / `streamlit_app.py`），但本仓库根已经有 `app/` 包。
放根目录会和包名冲突（`app.py` 会遮蔽 `app` 包，import 直接坏掉）。

**为什么 UI 不写在 `app/main.py` 里**：`main.py` 是 FastAPI 入口，
Streamlit 是**完全独立的进程模型**（每次交互重跑脚本）。混在一起，
两边的启动方式、依赖、错误处理全都会打架。
"""

from __future__ import annotations

__all__: list[str] = []
