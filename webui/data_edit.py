import os
import csv
import argparse
import gradio as gr
from typing import List

"""简洁 CSV WebUI（两列：wav_path, text）
- 启动参数指定 CSV，页面即加载
- 表格编辑；筛选仅作用于 text（不区分大小写），替换仅作用于 text（区分大小写）
- 点击“保存”写回 CSV（首行写入表头 wav_path,text）
"""

# 全局状态
g_csv_path: str = "dataset/audio_list/list.csv"
g_rows_all: List[List[str]] = []          # 全量数据（二维：[[wav_path, text], ...]）
g_view_map: List[int] = []                # 视图 -> 全量 索引映射
g_filter_desc: str = ""                  # 当前筛选描述
HEADERS = ["wav_path", "text"]
N_COLS = len(HEADERS)
TOAST_DURATION = 3


def _normalize_row(row: List[str], n_cols: int) -> List[str]:
    row = list(row or [])
    if len(row) < n_cols:
        row += [""] * (n_cols - len(row))
    elif len(row) > n_cols:
        row = row[:n_cols]
    return row


def _apply_view_rows_to_all(view_rows: List[List[str]]):
    global g_rows_all, g_view_map
    n_cols = N_COLS
    for i, vr in enumerate(view_rows or []):
        if i >= len(g_view_map):
            break
        all_idx = g_view_map[i]
        if 0 <= all_idx < len(g_rows_all):
            g_rows_all[all_idx] = _normalize_row(vr, n_cols)


def _to_rows(value) -> List[List[str]]:
    """将表格输入统一转为 List[List[str]]。"""
    if value is None:
        return []
    # pandas.DataFrame 兼容
    if hasattr(value, "values") and hasattr(value, "columns"):
        try:
            return [[str(c) for c in row] for row in value.values.tolist()]
        except Exception:
            try:
                import numpy as np  # noqa: F401
                return value.to_numpy().astype(str).tolist()  # type: ignore
            except Exception:
                pass
    # 普通 list
    if isinstance(value, list):
        out: List[List[str]] = []
        for r in value:
            if isinstance(r, (list, tuple)):
                out.append([str(c) for c in r])
            else:
                out.append([str(r)])
        return out
    return []


def _dataframe_update() -> gr.update:
    global g_rows_all, g_view_map, g_filter_desc
    view_rows = [g_rows_all[idx] for idx in g_view_map if 0 <= idx < len(g_rows_all)]
    subtitle = f"共 {len(g_rows_all)} 行；当前视图 {len(view_rows)} 行"
    if g_filter_desc:
        subtitle += f"（{g_filter_desc}）"
    n_cols = N_COLS
    return gr.update(
        headers=HEADERS,
        value=view_rows,
        label=subtitle,
        col_count=n_cols,
        row_count=(len(view_rows), "dynamic"),
    )


def _load_csv(path: str) -> List[List[str]]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    rows: List[List[str]] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)

        # 读取首行用于判断是否为表头（忽略大小写与 BOM）
        def _is_header(r: List[str]) -> bool:
            if not r:
                return False
            c0 = (r[0] or "").strip().lower().lstrip("\ufeff")
            c1 = (r[1] if len(r) > 1 else "").strip().lower()
            return c0 == "wav_path" and c1 == "text"

        first = next(reader, None)
        # 如果首行为表头，则跳过；否则将其当作数据行处理
        if first is not None and not _is_header([str(c) for c in first]):
            wav = str(first[0]) if len(first) >= 1 else ""
            txt = str(first[1]) if len(first) >= 2 else ""
            rows.append([wav, txt])

        # 继续处理剩余行
        for r in reader:
            r = [str(c) for c in r]
            wav = r[0] if len(r) >= 1 else ""
            txt = r[1] if len(r) >= 2 else ""
            rows.append([wav, txt])
    return rows


def _save_csv(path: str):
    global g_rows_all
    if not path:
        raise ValueError("保存路径为空")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(HEADERS)
        # 写数据行（两列）
        for r in g_rows_all:
            writer.writerow(_normalize_row(r, N_COLS))


def ui_save(df_value: List[List[str]]):
    """保存：先将视图改动写回全量，再落盘。"""
    global g_csv_path
    try:
        _apply_view_rows_to_all(_to_rows(df_value))
        _save_csv(g_csv_path)
        gr.Info(f"已保存至: {g_csv_path}", duration=TOAST_DURATION)
    except Exception as e:
        gr.Error(f"保存失败: {e}", duration=TOAST_DURATION)


def ui_filter(df_value: List[List[str]], query: str):
    """筛选：仅针对 text 列（不区分大小写）。"""
    global g_rows_all, g_view_map, g_filter_desc
    _apply_view_rows_to_all(_to_rows(df_value))

    q = (query or "").strip()
    if not q:
        g_view_map = list(range(len(g_rows_all)))
        g_filter_desc = ""
        return _dataframe_update()

    def _match(cell: str) -> bool:
        a = str(cell or "")
        return q.lower() in a.lower()

    new_map: List[int] = []
    for i, row in enumerate(g_rows_all):
        text_cell = row[1] if len(row) > 1 else ""
        if _match(text_cell):
            new_map.append(i)
    g_view_map = new_map
    g_filter_desc = f"筛选: text 包含 '{q}'，匹配 {len(new_map)} 行"
    return _dataframe_update()


def ui_reset_filter():
    global g_rows_all, g_view_map, g_filter_desc
    g_view_map = list(range(len(g_rows_all)))
    g_filter_desc = ""
    return _dataframe_update()


def ui_replace(df_value: List[List[str]], find: str, repl: str):
    """替换 text 列（全量）。"""
    global g_rows_all
    _apply_view_rows_to_all(_to_rows(df_value))

    find = str(find or "")
    repl = str(repl or "")
    if find == "":
        gr.Warning("查找内容为空，不执行替换", duration=TOAST_DURATION)
        return _dataframe_update()

    count = 0
    for i in range(len(g_rows_all)):
        if not (0 <= i < len(g_rows_all)):
            continue
        row = g_rows_all[i]
        # 仅替换 text 列
        if len(row) < N_COLS:
            row.extend([""] * (N_COLS - len(row)))
        cell = str(row[1])
        if find in cell:
            count += cell.count(find)
            row[1] = cell.replace(find, repl)
    gr.Info(f"替换完成：共替换 {count} 处", duration=TOAST_DURATION)
    return _dataframe_update()

def launch(port: int = 7861):
    with gr.Blocks(title="CSV 数据编辑器") as demo:
        gr.Markdown("## CSV 数据编辑器")

        # 顶部两行布局：第一行按钮（筛选/清除筛选/替换/清除内容），第二行输入（查询/查找/替换）
        with gr.Column():
            with gr.Row():
                btn_filter = gr.Button("应用筛选")
                btn_reset_filter = gr.Button("清除筛选")
                btn_replace = gr.Button("替换全部")
                btn_clear_replace = gr.Button("清除内容", variant="secondary")
            with gr.Row():
                tb_query = gr.Textbox(label="关键词筛选", scale=1)
                tb_find = gr.Textbox(label="查找", scale=1)
                tb_repl = gr.Textbox(label="替换为", scale=1)

        # 表格
        df = gr.Dataframe(
            headers=HEADERS,
            value=[],
            row_count=(0, "dynamic"),
            col_count=N_COLS,
            interactive=True,
            label="未加载",
        )

        with gr.Row():
            btn_save = gr.Button("保存", variant="primary", scale=1)

        # 事件绑定
        btn_save.click(ui_save, inputs=[df], outputs=[])

        btn_filter.click(ui_filter, inputs=[df, tb_query], outputs=[df])
        btn_reset_filter.click(ui_reset_filter, inputs=[], outputs=[df])
        btn_replace.click(ui_replace, inputs=[df, tb_find, tb_repl], outputs=[df])
        
        # 清除替换/查询输入内容
        def ui_clear_replace_fields():
            return gr.update(value=""), gr.update(value=""), gr.update(value="")
        btn_clear_replace.click(ui_clear_replace_fields, inputs=[], outputs=[tb_query, tb_find, tb_repl])


        # 初次加载：根据已加载的 CSV 初始化表格
        demo.load(_dataframe_update, inputs=[], outputs=[df])

    demo.launch(server_port=port)


def init_from_args(load_file: str):
    """根据命令行参数初始化全局数据并预加载 CSV。"""
    global g_csv_path, g_rows_all, g_view_map, g_filter_desc
    if not load_file:
        raise ValueError("必须通过 --load_file 指定要加载的 CSV 文件")
    rows = _load_csv(load_file)
    g_csv_path = load_file
    g_rows_all = rows
    g_view_map = list(range(len(g_rows_all)))
    g_filter_desc = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV 数据编辑器（wav_path,text）")
    parser.add_argument('--load_file', required=True, help='要加载的 CSV 文件路径')
    parser.add_argument('--server_port', type=int, default=7861, help='WebUI 端口，默认 7861')
    args = parser.parse_args()

    init_from_args(args.load_file)
    launch(port=args.server_port)
