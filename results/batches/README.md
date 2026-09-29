# 批次汇总目录

`scripts/run_all_cases.py` 每次完成后会在 `batch_YYYYMMDD_HHMMSS/` 中生成：

- `batch_summary.json`：完整的案例计数、半径选择、结果路径和方法限制；
- `batch_summary.csv`：一行一个案例，可直接用 Excel 打开比较。

批次目录是本地生成结果，默认不提交 GitHub。每次批处理建立新目录，不覆盖或删除旧批次。
