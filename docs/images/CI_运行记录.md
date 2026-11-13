# CI 运行记录（ent-doc-qa）

> 这是**可核对的事实记录**，不是截图。截图需要打开 github.com 的网页，本机浏览器打不开该站点（只能访问 api.github.com），
> 因此改用 README 上的官方 CI badge（随每次运行实时更新）+ 本文件记录 run ID / 链接 / 结论。
> 任何一条都可以点开核对。

## 最近一次运行（通过）

| 项 | 值 |
|---|---|
| 工作流 | CI（`.github/workflows/ci.yml`） |
| run ID | `43412553027` |
| 触发 | push 到 `main`（提交 `42e918f`） |
| 时间 | 2026-11-13T10:53:03Z |
| 结论 | **success** |
| 链接 | https://github.com/qie0403/ent-doc-qa/actions/runs/43412553027 |
| 矩阵 | Python 3.11 / 3.13 两个 job 均通过 |
| 覆盖 | `pytest tests/ -v --cov=src`，本地同版本 31 passed |

## 前一次运行（通过）

| 项 | 值 |
|---|---|
| run ID | `40110472116` |
| 时间 | 2026-10-24T16:12:21Z |
| 结论 | success |
| 链接 | https://github.com/qie0403/ent-doc-qa/actions/runs/40110472116 |

## 本地复现方式

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=src
```
