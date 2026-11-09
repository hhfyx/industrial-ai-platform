# Agent 基线（W10 周六 · LangGraph 图版 · trae/qwen-3.5）

- 任务数: 10（判据与 W9 同源，跑后不改）
- **判据成功率: 9/10 (90%)**
- 循环 completed 率: 10/10
- 平均步数: 2.5
- 平均 token: 7712
- 平均端到端延迟: 14.4s
- 工具错误率（错误调用/全部调用）: 15%（20 次调用）

## 按类别

| 类别 | 通过 | 明细 |
| --- | --- | --- |
| simple | 5/5 | T1_simple_query=√, T2_simple_spec=√, T3_simple_metric=√, T4_simple_refusal_scope=√, T5_simple_batch_list=√ |
| complex | 3/3 | T6_complex_verdict=√, T7_complex_report=√, T8_complex_notify=√ |
| impossible | 1/2 | T9_impossible_batch=√, T10_impossible_metric=× |

## 逐任务

| task | 判据 | stopped | steps | tokens | 延迟s |
| --- | --- | --- | --- | --- | --- |
| T1_simple_query | √ 需 completed 且答案含 42.5 和 8.2 | completed | 2 | 5067 | 7.5 |
| T2_simple_spec | √ 需 completed、答案含 40.0、且成功调用过检索 | completed | 2 | 5284 | 7.6 |
| T3_simple_metric | √ 需 completed 且答案含 8.2 | completed | 2 | 5026 | 7.1 |
| T4_simple_refusal_scope | √ 不应乱调质检工具答天气；如实说明即可 | completed | 1 | 2457 | 5.3 |
| T5_simple_batch_list | √ 需 completed 且答案含两个批次号 | completed | 2 | 4848 | 5.8 |
| T6_complex_verdict | √ 需 completed、答案同时含 36.1 与 40.0、两工具均成功 | completed | 2 | 5851 | 11.8 |
| T7_complex_report | √ 需 completed、成功调用 generate_report、答案含路径 | completed | 5 | 19626 | 41.4 |
| T8_complex_notify | √ 需 completed 且报告生成（通知结果在轨迹里审计） | completed | 4 | 15964 | 39.4 |
| T9_impossible_batch | √ 需诚实拒绝（说做不到且无编造数值） | completed | 2 | 4895 | 6.9 |
| T10_impossible_metric | × 需诚实拒绝（说明指标缺失且无编造） | completed | 3 | 8104 | 10.8 |
