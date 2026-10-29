# 容器化验收记录（W3 收尾）

> 2026-11-08。`docker compose up --build` 后的实测结果。
> **这份记录存在的理由**：容器里跑出来的 `degraded: true` 有**两个不同原因**，
> 不写清楚就会在下周被误判成"降级逻辑有 bug"。

---

## 一、验收结论

| 项 | 结果 |
| --- | --- |
| `docker compose up --build` | ✅ 构建并启动成功 |
| 容器健康检查 | ✅ `Up (healthy)`，`/health` 返回 `{"status":"ok"}` |
| `/api/v1/extract` 返回 200 | ✅ |
| `degraded` 标记出现 | ✅ `degraded: true` |
| 规则版抽取结果正确 | ✅ `合同编号=HT-2026-0042`、`签署日期=2026-03-15` |
| 契约严格性（多传一个键 → 422） | ✅ （见 `scripts/verify_degraded.py`） |

**`degraded: true` 是预期行为，不是缺陷。** 原因见下节。

---

## 二、为什么容器里必然降级（两个独立原因）

### 原因 1：`127.0.0.1` 在容器里指的是容器自己

`.env` 里的 `LLM_BASE_URL=http://127.0.0.1:8326/v1` 在**宿主机**上是对的，
但在**容器内**，`127.0.0.1` 是容器自己的 loopback，不是宿主机网关。

实测（容器内执行）：

```
127.0.0.1:8326        → ConnectionRefusedError
host.docker.internal:8326 → CONNECTED
```

### 原因 2：网关要求登录态，容器内的调用不被认可

把 `LLM_BASE_URL` 改成 `host.docker.internal:8326` 后，网络通了，但仍降级。
容器日志给出真实原因：

```
LLM 抽取失败，降级规则版：
RetryExhaustedError: 重试 1 次后仍失败：
InternalServerError: Error code: 502 -
{'error': {'message': '未配置凭据，请先登录', 'type': 'server_error', 'code': 'UNKNOWN'}}
```

**本机网关（端口 8326）是 DeepSeek Harness 进程内起的**，
它的鉴权跟着宿主进程的登录态走。容器是另一个进程、另一种身份，
即使网络可达，网关也不认。

> 这两个原因**互相独立**：只修网络不够，还要解决鉴权。
> 所以"改成 host.docker.internal 就好了"是个不完整的诊断。

---

## 三、为什么不去"修"它

**不修**，三个理由：

1. **降级路径本来就是设计要验证的东西。** `degraded: true` 是 api-spec v1.1 的
   契约字段，它的作用就是"让省钱的代价可见"。容器里跑出它，说明这条路径**真的能用**。
2. **改 `.env` 去迁就容器，会掩盖配置错误的信号。**
   如果 compose 里硬写 `host.docker.internal`，那"宿主机上直跑"就会开始报错 ——
   同一个 `.env` 不能在两种环境里同时正确，**这是个必须知道的知识点，不是要抹平的问题**。
3. **鉴权问题是环境限制，不是代码缺陷。** 要真正在容器里跑通 LLM，
   需要一个独立的、可被任意进程使用的 API key（而不是宿主机进程的登录态）。
   那属于部署配置，不在本周的验收范围内。

---

## 四、已做的改动（只改文档，不改行为）

- [`.env.example`](../.env.example) 里 `LLM_BASE_URL` 上方补了容器场景的警告，
  写明两种环境该分别用什么值。

**没有改** `docker-compose.yml`、没有改 `.env`、没有改代码逻辑 ——
行为保持"错的配置会以 `degraded:true` 可见地暴露"。

---

## 五、复跑

```powershell
cd D:\git\网页\industrial-ai-platform

# 启动（需要 Docker Desktop 在跑）
docker compose up -d --build
docker ps --format "table {{.Names}}\t{{.Status}}"

# 验健康
curl.exe http://127.0.0.1:8000/health

# 验降级契约（容器里必然是 degraded:true）
.\.venv\Scripts\python.exe -X utf8 scripts\verify_degraded.py

# 看降级原因
docker compose logs --tail 30 api

# 清理
docker compose down
```

> `scripts/verify_degraded.py` 默认打的是 8011 端口（uvicorn 直跑的场景）。
> 验容器时把 `BASE` 改成 `http://127.0.0.1:8000`。

---

## 六、本周这条验收的诚实边界

**我验到的是**："容器能起、健康、接口 200、降级契约字段正确出现、规则版结果正确"。

**我没验到的是**："容器里的 LLM 路径能跑通"。

后者需要两件事同时满足：
1. `LLM_BASE_URL` 指向宿主机可达地址（`host.docker.internal`）
2. 网关接受来自容器的鉴权

第 2 条在当前本机网关上是**做不到的**（它绑定宿主进程登录态）。
所以"容器里跑 LLM 抽取"这条**本周未验证**，下周若要上真实容器部署必须先解决。
