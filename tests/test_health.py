from fastapi.testclient import TestClient

from app import __version__
from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_is_available() -> None:
    """版本号从 `app.__version__` 读，不写死。

    写死 `"0.1.0"` 的话，每次升版本这条测试都会红一次 —— 而它想守的是
    "openapi 能生成"，不是"版本号没变"。测试里硬编码的常量，最容易变成噪音。
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == __version__
