"""容器内验证 session 落库可查（W8 周日 14:30 格）。

运行：docker cp 进容器后 docker exec industrial-ai-api python /tmp/check_sessions.py
"""

import json
import sqlite3

conn = sqlite3.connect("/app/data/sessions/sessions.sqlite")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", tables)
n = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
print("turns rows:", n)
rows = conn.execute(
    "SELECT session_id, seq, role, substr(content, 1, 40), rewrite_of FROM turns ORDER BY seq LIMIT 5"
).fetchall()
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
conn.close()
