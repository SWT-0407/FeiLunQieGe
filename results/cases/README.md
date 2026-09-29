# 工件案例结果目录

这里按学长提供的 12 个案例名称分目录保存后续运行结果。原始 OBJ/PLY 不复制到仓库；每个案例的运行结果放在 `run_YYYYMMDD_HHMMSS/` 下，并且每次运行建立新目录，不覆盖旧结果。

当前代码已把 `case_name` 作为结果目录的一部分。12 个案例均已完成一套本地批处理结果；每套结果包含精确可行中心 OBJ、交互式 HTML、JSON 参数摘要、两个 MeshLab 显示代理 OBJ/MTL 和三张 PNG 辅助图。新的运行仍会建立时间戳目录，不覆盖旧结果。

批处理命令为 `python scripts/run_all_cases.py --config configs/default.json --continue-on-error`。默认在内存中把工件及 `root_line` 一起归一化为包围盒最长边为 1；`r` 通过候选半径扫描自动选择，也可以在 `radius_selection.case_overrides` 中按案例指定。原始模型位于仓库父目录，不复制进 Git；生成结果因体积较大默认不提交 GitHub。
