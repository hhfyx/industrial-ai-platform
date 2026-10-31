"""部署配置测试：免费层降级必须**有代码**，不能只写在文档里。

## 为什么要测这个

`docs/deploy-notes.md` 里写了"免费层内存不够，用 hashing 降级"。
但**文档会被忘掉，代码不会**。如果降级只存在于文档：
下一个人（或三个月后的我）直接改 `requirements-deploy.txt` 加回 torch，
应用会在线上静默 OOM —— 而 OOM 的表现是"应用隔几分钟重启一次"，
不是一条能看懂的错误。

所以三件事必须由测试守着：

1. 平台标记存在时，默认 embedder 是 hashing
2. 显式配置（环境变量 / 参数）能覆盖平台默认
3. 缺 torch 时 `bge.embed()` 抛出的是**人话**，不是裸 `ModuleNotFoundError`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.deploy_config import (
    EMBEDDER_ENV,
    FREE_TIER_DEFAULT,
    describe_runtime,
    looks_like_free_hosting,
    resolve_embedder_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================ 1. 降级选择


def test_free_tier_default_is_hashing() -> None:
    """免费层默认必须是 hashing。

    实测依据（`scripts/measure_memory.py`）：
      - 只 import torch      ~250 MB
      - + bge-small-zh        **561 MB**
      - 免费层上限             1 GB
    再加 Streamlit + Chroma，bge 路径必然撞墙。
    """
    assert FREE_TIER_DEFAULT == "hashing"


@pytest.mark.parametrize(
    "marker",
    ["IS_STREAMLIT_CLOUD", "STREAMLIT_SHARING_MODE", "K_SERVICE", "DYNO"],
)
def test_free_hosting_markers_are_recognized(marker: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """**这四个变量名是硬编码的契约**。

    平台改名或换实现时，检测会**静默失效** —— 退回 bge → 又 OOM。
    所以这里逐个别名断言。改这些变量名必须同时改这条测试，
    那是一次"有意识的决定"，而不是一次意外。
    """
    monkeypatch.delenv(EMBEDDER_ENV, raising=False)
    monkeypatch.setenv(marker, "1")
    assert looks_like_free_hosting() is True
    assert resolve_embedder_name() == FREE_TIER_DEFAULT


def test_no_marker_means_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有平台标记时**不算**免费层。

    这条防止"本地跑也降级"—— 本地有 16 GB 内存，降级纯属自损。
    特意检查：不接受 `STREAMLIT_SERVER_*` 这类本地也有的变量。
    """
    for marker in ("IS_STREAMLIT_CLOUD", "STREAMLIT_SHARING_MODE", "K_SERVICE", "DYNO"):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "8501")  # 本地也有，不该触发
    assert looks_like_free_hosting() is False


def test_explicit_env_overrides_platform_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量优先于平台推断 —— 付费层想用 bge 时靠它。"""
    monkeypatch.setenv("IS_STREAMLIT_CLOUD", "1")
    monkeypatch.setenv(EMBEDDER_ENV, "bge-small-zh")
    assert resolve_embedder_name() == "bge-small-zh"


def test_explicit_argument_beats_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式参数优先级最高。"""
    monkeypatch.setenv(EMBEDDER_ENV, "hashing")
    assert resolve_embedder_name("bge-small-zh") == "bge-small-zh"


def test_describe_runtime_gives_a_visible_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """降级必须给出**可显示**的提示文本。

    **悄悄降级等于骗用户**：他以为在做语义检索，实际只有字面匹配。
    所以 `describe_runtime()` 不只是返回布尔值，还返回一句人话，
    UI 直接把它显示成 warning。
    """
    monkeypatch.setenv("IS_STREAMLIT_CLOUD", "1")
    runtime = describe_runtime()

    assert runtime["free_hosting_detected"] is True
    assert runtime["notice"], "降级时必须有一句能显示的提示"
    assert "hashing" in str(runtime["notice"])
    assert "561" in str(runtime["notice"]), "提示里要给内存数字，否则理由不成立"

    # 本地时不显示提示（避免狼来了）
    monkeypatch.delenv("IS_STREAMLIT_CLOUD")
    assert describe_runtime()["notice"] is None


# ============================================================ 2. 缺依赖时的错误信息


def test_bge_missing_dependency_raises_readable_error() -> None:
    """**缺 sentence-transformers 时必须报人话**。

    在子进程里屏蔽 torch / sentence_transformers，模拟 Streamlit Cloud
    按 `requirements-deploy.txt` 装依赖（故意不含 torch）。

    裸 `ModuleNotFoundError: No module named 'sentence_transformers'`
    会让运维去查"是不是包冲突"，而真相是"你选了一个这个环境装不下的模型"——
    这两件事的应对完全不同（前者修依赖，后者换实现）。
    """
    script = """
import sys
class Blocker:
    BLOCKED = ("torch", "sentence_transformers")
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"[simulated] no module named {fullname}")
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, ".")
from app.rag.embedding import get_embedder
e = get_embedder("bge-small-zh")
try:
    e.embed(["x"])
except RuntimeError as exc:
    msg = str(exc)
    assert "sentence-transformers" in msg, msg
    assert "561" in msg, "错误里要给内存数字"
    assert "hashing" in msg, "错误里要给可行的替代方案"
    print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"子进程失败：{result.stderr[-800:]}"
    assert "OK" in result.stdout


def test_hashing_works_without_torch() -> None:
    """**降级路径要真的能跑通整条链路**，不只是"构造成功"。

    在子进程里屏蔽 torch，然后索引 + 检索各跑一次。
    """

    script = """
import sys
class Blocker:
    BLOCKED = ("torch", "sentence_transformers")
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"[simulated] no module named {fullname}")
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, ".")

from app.rag.rag_availability import bge_available
assert bge_available() is False, "屏蔽失效了，测试没意义"

from app.rag.embedding import get_embedder
from app.rag.pipeline import RagPipeline
embedder = get_embedder("hashing")
pipeline = RagPipeline(embedder=embedder, top_k=2)
report = pipeline.index_text("违约金按日万分之五支付。", source="合同")
assert report.ok, report.to_dict()
hits = pipeline.retrieve("违约金").hits
assert len(hits) >= 1
print("OK chunks=", report.chunks, "hits=", len(hits))
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"降级路径跑不通：{result.stderr[-800:]}"
    assert "OK" in result.stdout


# ============================================================ 3. 部署文件自洽


def test_deploy_requirements_excludes_torch() -> None:
    """`requirements-deploy.txt` **不能含 torch / sentence-transformers**。

    这是免费层能装完、能跑起来的前提。有人"顺手补上缺的包"会让线上 OOM ——
    而那份文件里已经写了理由，这条测试是给"不读注释的人"看的。
    """
    content = (PROJECT_ROOT / "requirements-deploy.txt").read_text(encoding="utf-8")
    # 只看非注释行
    lines = [
        ln.strip() for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    package_names = {ln.split("==")[0].split(">=")[0].lower() for ln in lines}

    assert "torch" not in package_names, (
        "requirements-deploy.txt 不能含 torch —— 免费层装不下（详见文件头注释）"
    )
    assert "sentence-transformers" not in package_names


def test_deploy_requirements_has_the_core_stack() -> None:
    """部署依赖必须覆盖 UI 与 RAG 核心，否则线上直接 ImportError。"""
    content = (PROJECT_ROOT / "requirements-deploy.txt").read_text(encoding="utf-8")
    lines = [
        ln.strip() for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    names = {ln.split("==")[0].split(">=")[0].lower() for ln in lines}

    required = {
        "streamlit",  # UI
        "chromadb",  # 向量库
        "pypdf",  # PDF 解析
        "pdfplumber",  # 表格
        "python-docx",  # DOCX
        "jinja2",  # qa 模板
        "pydantic",  # schema
        "pydantic-settings",  # 配置
        "openai",  # LLM 客户端
    }
    missing = required - names
    assert not missing, f"部署依赖缺少：{missing}"


def test_streamlit_cloud_config_exists() -> None:
    """Streamlit Cloud 靠仓库根的 `.streamlit/config.toml` 与入口文件识别应用。

    少了它们，部署会"找不到 app"，而报错信息不会告诉你要补哪个文件。
    """
    assert (PROJECT_ROOT / ".streamlit" / "config.toml").exists()
    assert (PROJECT_ROOT / "app" / "ui" / "streamlit_app.py").exists()


def test_streamlit_config_has_no_invalid_keys() -> None:
    """**配置项位置写错会静默失效**。

    实测踩到：把 `maxUploadSize` 写到 `[client]` 段，Streamlit 只打一行警告
    然后**用默认值继续跑** —— 于是"我明明调了上限"和"还是传不上去"同时成立。
    正确位置是 `[server]`。
    """
    import tomllib

    config_text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    config = tomllib.loads(config_text)

    assert "server" in config
    assert config["server"].get("maxUploadSize") == 200
    # `[client]` 段不该有 maxUploadSize
    assert "maxUploadSize" not in config.get("client", {}), (
        "maxUploadSize 属于 [server] 段；写在 [client] 会被静默忽略"
    )


# ============================================================ 4. memory 实测脚本可跑


def test_memory_measure_script_is_documented() -> None:
    """内存数字必须有**可复跑**的脚本，否则它是"某个人记得的一个数"。

    这条只检查脚本存在且被文档引用 —— 真跑一次要 20 秒并加载模型，
    不适合放进单元测试。**跑一次是人做的事，忘记跑才是自动化该防的。**
    """
    script = PROJECT_ROOT / "scripts" / "measure_memory.py"
    assert script.exists(), "缺 scripts/measure_memory.py"

    notes = PROJECT_ROOT / "docs" / "deploy-notes.md"
    assert notes.exists(), "缺 docs/deploy-notes.md"
    assert "measure_memory.py" in notes.read_text(encoding="utf-8"), (
        "deploy-notes.md 必须指向那个可复跑的测内存脚本"
    )
