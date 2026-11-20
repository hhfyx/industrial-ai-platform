"""合同审批模块（企业业务侧，W14）。

分层纪律（docs/module-contract-approval.md §六）：

    state_machine.py / approval_rules.py   纯函数，**不 import 任何 DB**
    models.py / db.py                      ORM 与引擎
    service.py                             事务边界全部在这里
    repository.py                          查询封装（报表 SQL）
    reports.py                             CSV 导出

app/api/v2/contracts.py + reports.py 是对外的 5 类端点；
错误一律抛 app.core.errors.AppError，走统一错误体出口。
"""

from __future__ import annotations
