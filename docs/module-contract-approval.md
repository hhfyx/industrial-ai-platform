# 合同审批模块（企业业务侧）· 建表与接口设计

> **状态（2026-11-20 收口）：验收 6 条全部达成** ✅
> | # | 验收结果 |
> | --- | --- |
> | 1 | ✅ MySQL 8 容器 healthy（3307），5 表由 `alembic upgrade head` 创建，`SHOW TABLES` 逐一核对 |
> | 2 | ✅ 21 条状态机测试绿（合法迁移 / 非法迁移 409） |
> | 3 | ✅ 真实 API 走完整链路：draft → submit(pending) → approve(v2, 仍 pending=一级) → approve(v3, approved=两级) |
> | 4 | ✅ **20 并发 approve：恰好 1 成功 + 19×409 + 日志只多 1 条**（MySQL 活体测试，非 skip） |
> | 5 | ✅ 报表 JSON 聚合 + CSV 导出实测出数（Content-Type: text/csv; charset=utf-8） |
> | 6 | ✅ `approval_logs` 留痕（submit/approve 各一条，含 actor/comment/状态迁移） |
>
> 乐观锁行为实测：stale `expected_version` 重放 → HTTP 409 拒绝。
> Streamlit tab 未做（"不做"清单里的项，按原计划搁置）。

> **为什么做这个**：目标 JD（AI 软件开发工程师 / 企业信息化）第 6 条要
> 「MySQL/PostgreSQL + 基础 SQL + 表结构与接口基本设计」，第 9 条要「愿意学
> 企业业务流程、审批流程和报表生成逻辑」。
> 现有项目在数据库上只有「SQLite 只读查询工具」（安全视角），**没有业务侧建表、
> 没有事务、没有审批状态机、没有报表导出**。本模块补的正是这块。
>
> **不另起仓库**：沿用 `industrial-ai-platform` 的 API 契约、错误体、鉴权、
> request_id、pytest 纪律。新代码只落在一个新目录 + 一个 MySQL 容器。

---

## 一、验收标准（做完这 6 条就算完成）

| # | 验收项 | 怎么验 |
| --- | --- | --- |
| 1 | MySQL 8 在 compose 里起得来，5 张表由迁移脚本创建 | `docker compose -f docker-compose.biz.yml up -d mysql` + `alembic upgrade head` |
| 2 | 合同 CRUD + 状态机（5 态、7 条合法迁移、非法迁移当场 409） | `pytest tests/test_contract_state.py` |
| 3 | 两级审批（金额阈值路由：< 10 万一级，≥ 10 万两级） | 造两条合同，走完整审批链 |
| 4 | 并发安全：两个审批人同时点通过，只有一个生效 | `pytest tests/test_approve_concurrency.py`（并发 20 次） |
| 5 | 报表：按月/按客户汇总，导出 CSV | `GET /api/v2/reports/contracts.csv` |
| 6 | 审计：每次状态变更留痕（谁、何时、从什么态到什么态、备注） | 查 `approval_logs` 表 |

**不做**：前端页面（用现有 Streamlit 加一个 tab 即可）、权限体系（复用 API Key）、
多租户、附件上传、电子签章。

---

## 二、数据模型（5 张表）

### 2.1 DDL

```sql
-- 1. 客户
CREATE TABLE customers (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  name          VARCHAR(128) NOT NULL,
  tax_no        VARCHAR(32)  NULL,            -- 纳税人识别号
  contact       VARCHAR(64)  NULL,
  created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  UNIQUE KEY uk_customers_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 合同（主表）
CREATE TABLE contracts (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  contract_no   VARCHAR(32)  NOT NULL,        -- 业务单号，如 HT-20261201-001
  customer_id   BIGINT UNSIGNED NOT NULL,
  title         VARCHAR(200) NOT NULL,
  amount_cents  BIGINT       NOT NULL,        -- 金额用「分」存整数，不用 DECIMAL/FLOAT
  status        ENUM('draft','pending','approved','rejected','cancelled')
                NOT NULL DEFAULT 'draft',
  version       INT UNSIGNED NOT NULL DEFAULT 0,  -- 乐观锁
  created_by    VARCHAR(64)  NOT NULL,
  created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  UNIQUE KEY uk_contracts_no (contract_no),
  KEY idx_contracts_customer (customer_id),
  KEY idx_contracts_status_created (status, created_at),
  CONSTRAINT fk_contracts_customer FOREIGN KEY (customer_id) REFERENCES customers(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 审批节点（一张合同要经过几个节点，谁审）
CREATE TABLE approval_nodes (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  contract_id   BIGINT UNSIGNED NOT NULL,
  seq           TINYINT UNSIGNED NOT NULL,    -- 1, 2, ...
  role          VARCHAR(32) NOT NULL,         -- manager / director
  status        ENUM('waiting','approved','rejected','skipped') NOT NULL DEFAULT 'waiting',
  approver      VARCHAR(64)  NULL,            -- 实际审批人
  decided_at    DATETIME(3)  NULL,
  UNIQUE KEY uk_nodes_contract_seq (contract_id, seq),
  CONSTRAINT fk_nodes_contract FOREIGN KEY (contract_id) REFERENCES contracts(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 审批日志（只增不改，审计用）
CREATE TABLE approval_logs (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  contract_id   BIGINT UNSIGNED NOT NULL,
  from_status   VARCHAR(16) NOT NULL,
  to_status     VARCHAR(16) NOT NULL,
  action        VARCHAR(16) NOT NULL,         -- submit/approve/reject/cancel
  actor         VARCHAR(64) NOT NULL,
  comment       VARCHAR(500) NULL,
  created_at    DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  KEY idx_logs_contract (contract_id, created_at),
  CONSTRAINT fk_logs_contract FOREIGN KEY (contract_id) REFERENCES contracts(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. 幂等键（复用 W8 的投递幂等思想）
CREATE TABLE idempotency_keys (
  key_hash      CHAR(64) PRIMARY KEY,         -- sha256(key)
  scope         VARCHAR(32) NOT NULL,         -- 如 approve:123
  response_json JSON NULL,
  created_at    DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  KEY idx_idem_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 2.2 三个必须能讲清的设计决定

**① 金额为什么用 `amount_cents BIGINT` 而不是 `DECIMAL`/`FLOAT`**

`FLOAT` 有精度误差（0.1+0.2≠0.3），合同金额不能有误差。
`DECIMAL` 也行，但 Java/Python 两侧都要小心类型映射；**用「分」存整数**是最省心的做法，
展示层再除 100。**这条面试可以直接问：为什么不用 FLOAT？**

**② `version` 字段是干什么的**

乐观锁。两个审批人同时点「通过」，两人读到的都是 `version=0`，
先提交的 `UPDATE ... WHERE version=0` 成功（version→1），
后提交的 `WHERE version=0` 影响 0 行 → 返回 409「该合同已被处理」。
**不用悲观锁 `SELECT FOR UPDATE` 的原因**：审批是低频操作，乐观锁不需要长事务、
不会占连接，冲突时让用户重试即可。

**③ `approval_logs` 为什么单独一张表，而不是在 `contracts` 存个 `history` JSON**

审计要求是**只增不改、可独立查询**。塞进主表 JSON 会变成「改合同就等于改历史」，
而且没法按时间/操作人建索引。这张表只 INSERT，永远不 UPDATE/DELETE。

---

## 三、状态机（5 态 / 7 条合法迁移）

```
                 submit                approve(末节点)
   draft ───────────────► pending ──────────────────► approved
     │                      │  │
     │ cancel               │  └── reject ─────────► rejected
     ▼                      │
  cancelled                 └── cancel ─────────────► cancelled
```

| # | from → to | 触发 | 前置条件 | 副作用 |
| --- | --- | --- | --- | --- |
| 1 | draft → pending | submit | 金额>0、客户存在、节点已生成 | 生成 approval_nodes |
| 2 | pending → pending | approve(非末节点) | 当前节点是 seq 最小且 waiting | 该节点 approved，下一节点激活 |
| 3 | pending → approved | approve(末节点) | 同上，且无后续节点 | 全部节点 completed |
| 4 | pending → rejected | reject | 任一节点 | 余下节点 skipped |
| 5 | draft → cancelled | cancel | 创建人或管理员 | — |
| 6 | pending → cancelled | cancel | 创建人（未有人审批）| 余下节点 skipped |
| 7 | rejected → draft | revise | 创建人 | 重置节点、version+1 |

**非法迁移一律 `AppError("INVALID_STATE_TRANSITION", ..., status_code=409)`**，
并在 message 里写清「当前是 approved，不能 submit」。

---

## 四、API 契约（5 个端点）

沿用现有约定：`X-API-Key` 鉴权、`X-Request-ID` 回显、错误体走
`ErrorResponse{code, message, request_id, details}`（`app/core/errors.py` 已有）。

### 4.1 建客户 + 建合同

```http
POST /api/v2/customers
{ "name": "某某建工集团", "tax_no": "9133...", "contact": "张工 139..." }
→ 201 { "id": 1, "name": "...", }

POST /api/v2/contracts
{ "contract_no": "HT-20261201-001", "customer_id": 1,
  "title": "2026 年度预制构件供货合同", "amount_cents": 128000000 }
→ 201 { "id": 1, "status": "draft", "version": 0,
        "nodes": [{"seq":1,"role":"manager"},{"seq":2,"role":"director"}] }
```

**金额阈值路由**：`amount_cents < 10_000_000`（10 万元）→ 只生成 1 个节点（manager）；
`>= 10_000_000` → 生成 2 个节点（manager + director）。
**这条规则写在 `app/biz/approval_rules.py` 里，单独可测。**

### 4.2 提交 / 审批 / 驳回 / 撤回

```http
POST /api/v2/contracts/{id}/submit
POST /api/v2/contracts/{id}/approve
  { "comment": "金额与预算一致", "expected_version": 0 }
POST /api/v2/contracts/{id}/reject
  { "comment": "单价高于市场价 8%", "expected_version": 0 }
POST /api/v2/contracts/{id}/cancel
  { "comment": "客户取消订单", "expected_version": 0 }
```

**`expected_version` 是必须传的**。它让「读-改-写」的竞态在客户端就暴露出来，
而不是服务端悄悄覆盖。服务端校验：请求的 `expected_version != 表里 version` → 409。

### 4.3 查询与报表

```http
GET /api/v2/contracts?status=pending&customer_id=1&page=1&size=20
GET /api/v2/contracts/{id}              → 含 nodes 与 logs
GET /api/v2/reports/contracts           → JSON 汇总
GET /api/v2/reports/contracts.csv       → CSV 下载
```

报表 SQL（**这是 JD「写基础 SQL」的直接证据**）：

```sql
SELECT c.customer_id,
       cu.name                          AS customer_name,
       DATE_FORMAT(c.created_at, '%Y-%m') AS month,
       COUNT(*)                         AS cnt,
       SUM(c.amount_cents) / 100.0      AS amount_yuan,
       SUM(c.status = 'approved')       AS approved_cnt
FROM contracts c
JOIN customers cu ON cu.id = c.customer_id
WHERE c.created_at >= :start AND c.created_at < :end
GROUP BY c.customer_id, cu.name, month
ORDER BY amount_yuan DESC
LIMIT 100;
```

---

## 五、并发与事务（这块是面试区分度最高的地方）

### 5.1 审批的完整事务

```python
async with session.begin():  # 一个事务
    # 1. 行锁读（只锁这一行，不是整表）
    row = await session.get(Contract, cid, with_for_update=True)
    # 2. 校验版本（乐观锁 + 悲观锁双保险，实际二选一，注释里写清为什么）
    if row.version != expected_version:
        raise AppError("VERSION_CONFLICT", "该合同已被他人处理，请刷新后重试", status_code=409)
    # 3. 找当前待审节点
    node = await find_current_node(session, cid)
    # 4. 状态机校验（非法迁移在这里抛）
    next_status = transition(row.status, "approve", is_last=is_last_node)
    # 5. 落三张表
    node.status = "approved"
    node.approver = actor
    node.decided_at = now()
    if next_status == "approved":
        await skip_remaining_nodes(session, cid)
    row.status = next_status
    row.version += 1
    session.add(ApprovalLog(from_status=old, to_status=next_status, action="approve", actor=actor))
# 事务边界在这里闭合：要么三张表都成功，要么都不动
```

**必须能回答的三个追问**：

1. **为什么审批要在一个事务里改三张表？**
   否则会出现「节点已通过、主合同还是 pending」或「日志记了、状态没改」的脏数据。
2. **乐观锁和悲观锁你怎么选的？**
   用 `expected_version`（乐观），因为审批冲突概率低，不想让事务持有行锁等待用户思考。
   `with_for_update` 只是防止同一瞬间两个请求都通过版本校验（可去掉，见注释）。
3. **并发 20 个请求同时 approve，会怎样？**
   1 个成功，19 个返回 409。**测试要验证「approval_logs 只多 1 条」**，
   不是「19 个都失败」——因为失败可能是 500 而不是 409，那不算对。

### 5.2 幂等（复用 W8 投递幂等的思路）

审批接口接受可选头 `Idempotency-Key`。同一 key 重复提交 → 返回**首次的结果**（200），
不重复执行。用于前端网络重试、用户连点两次。

---

## 六、目录结构（放进现有项目）

```
app/biz/                          ← 新增，业务侧独立于 llm/
├── __init__.py
├── models.py                     SQLAlchemy 2.0 ORM（5 张表）
├── schemas.py                    Pydantic v2 请求/响应
├── state_machine.py              纯函数：transition(from, action) → to | raise
├── approval_rules.py             金额阈值 → 节点数（可单测，无 DB 依赖）
├── service.py                    事务边界都在这里
├── repository.py                 查询封装（报表 SQL 放这里）
└── reports.py                    CSV 导出

app/api/v2/
├── contracts.py                  ← 新增，5 个端点
└── reports.py                    ← 新增

migrations/                       ← Alembic
└── versions/0001_contract_approval.py

tests/
├── test_contract_state.py        状态机 7 条合法 + 非法迁移
├── test_approval_rules.py        金额阈值（纯函数，最快）
├── test_approve_concurrency.py   并发 20 次
├── test_reports.py               报表 SQL 与 CSV
└── test_biz_api.py               接口契约（错误体、401、409）

docker-compose.biz.yml            MySQL 8 + 现有 app
```

**分层纪律**：`state_machine.py` 和 `approval_rules.py` **不 import 任何 DB**，
所以它们是纯函数、毫秒级测试。`service.py` 才碰事务。这条分层是现有项目
「契约冻结」思路的延续。

---

## 七、测试策略（1386 条那套纪律的延续）

| 层 | 测什么 | 数量 | 特点 |
| --- | --- | --- | --- |
| 纯函数 | 状态机 7 条合法 + 至少 8 条非法；金额阈值 3 档 | ~15 | 无 DB，毫秒级 |
| 仓储 | 报表 SQL（月份边界、空结果、金额合计） | ~8 | 需 MySQL |
| 并发 | 20 并发 approve → 1 成功 19×409，日志只多 1 条 | 1 | 需 MySQL，最值钱 |
| 接口 | 201/409/401/422 契约形状、request_id 回显 | ~12 | TestClient |
| 幂等 | 同 key 两次 → 结果一致且只执行一次 | 2 | 需 MySQL |

**测试用 MySQL 而不是 SQLite**（重要，面试会问）：SQLite 与 MySQL 在
外键强制、`ENUM`、`NUMERIC` 精度、并发锁行为、`DATE_FORMAT` 上都不一样。
**用 SQLite 测「并发只有一个成功」根本测不出来**——它默认串行化写入。
测试夹具起一个 MySQL 容器，跑完清理。

---

## 八、两天排期（周末）

### Day 1（8h）

| 时段 | 任务 | 产出 |
| --- | --- | --- |
| 09:00–10:00 | MySQL 容器 + SQLAlchemy 接入 + Alembic 初始化 | `docker compose -f docker-compose.biz.yml up` 能连 |
| 10:00–12:00 | 5 张表 ORM + 首个迁移脚本 | `alembic upgrade head` 建表成功 |
| 13:30–15:00 | 状态机（纯函数）+ 金额路由 + 单元测试 | 23 条纯函数测试绿 |
| 15:00–17:00 | 建客户/建合同/提交 三个接口 | 能造出一条 pending 合同 |

### Day 2（8h）

| 时段 | 任务 | 产出 |
| --- | --- | --- |
| 09:00–10:30 | approve / reject / cancel（含事务 + 版本校验） | 审批链走得通 |
| 10:30–12:00 | 并发测试 + 幂等 | 20 并发测试绿 |
| 13:30–15:00 | 查询 + 报表 + CSV 导出 | 报表能出数 |
| 15:00–16:00 | 接进 Streamlit（加一个「合同审批」tab） | 可演示 |
| 16:00–17:00 | 写 README 章节 + 简历那一行 | 收口 |

**总代码量估计**：ORM ~250 行、service ~300 行、API ~200 行、测试 ~350 行、
迁移 ~80 行。**1200 行左右，两天够。**

---

## 九、这一块写进简历该怎么写

现有简历项目一可以加一行（或替换掉一条评测细节）：

> **企业业务侧（合同审批）**：MySQL 8 + SQLAlchemy 2.0 设计 5 表（客户/合同/
> 审批节点/审计日志/幂等键），实现 5 态审批状态机与金额阈值路由；
> 用 `version` 乐观锁 + 单事务改三表保证并发正确性（20 并发仅 1 成功、审计日志不重复）；
> 报表用 `GROUP BY` 月份聚合 + CSV 导出；全部 23 条纯函数测试 + 并发测试接 CI。

**技术栈那栏**加：`MySQL / SQLAlchemy 2.0 / Alembic`

**技能栏**加一条：

> **后端与数据**：FastAPI 分层（路由/服务/仓储）；MySQL 表结构与索引设计
> （外键、联合索引、ENUM、金额用分存整数）；事务边界与乐观锁；
> Alembic 迁移；报表聚合 SQL

---

## 十、做完之后，这个 JD 的缺口还剩什么

| JD 要求 | 做完本模块后 |
| --- | --- |
| 第 6 条 MySQL + 表结构 + 基础 SQL | ✅ **补齐**（还超出：事务/锁/迁移/报表） |
| 第 9 条 企业业务流程/审批/报表 | ✅ **补齐** |
| 第 5 条 前端 Vue/React/原生 | ⚠️ 仍靠 `static/index.html` 原生 JS + Streamlit |
| 第 8 条 AI 工具三意识 | ⚠️ 需**改简历写出来**（能力已有，只是没写） |
| 第 3/4 条 Python + FastAPI | ✅ 已强 |

**剩下两个缺口都不是「做项目」能补的**：第 5 条要么学 Vue（一周），
要么在简历里明确写「原生 HTML/CSS/JS 手写，未用框架」；第 8 条纯改简历。
