import argparse
import gradio as gr

# 复用现有模块逻辑（不改动原文件）
import subfix_webui as sf
import dataEdit_webui as de


def _patch_gr_toast_duration(duration: float):
    """统一 gr.Info/Warning/Error 的默认显示时长。"""
    try:
        _orig_info = gr.Info
        _orig_warn = gr.Warning
        _orig_err = gr.Error

        def _wrap_info(message, *args, **kwargs):
            kwargs.setdefault("duration", duration)
            return _orig_info(message, *args, **kwargs)

        def _wrap_warn(message, *args, **kwargs):
            kwargs.setdefault("duration", duration)
            return _orig_warn(message, *args, **kwargs)

        def _wrap_err(message, *args, **kwargs):
            kwargs.setdefault("duration", duration)
            return _orig_err(message, *args, **kwargs)

        gr.Info = _wrap_info  # type: ignore
        gr.Warning = _wrap_warn  # type: ignore
        gr.Error = _wrap_err  # type: ignore
    except Exception:
        pass


def build_app(args):
    # 初始化两个子模块的全局状态
    sf.set_global(args.load_file, args.g_batch, args.webui_language)
    de.init_from_args(args.load_file)

    # 统一全局 Toast 显示时长
    # 1) SubFix 内部使用其模块常量；这里覆盖为统一值
    try:
        sf.TOAST_DURATION = args.toast_duration  # type: ignore
    except Exception:
        pass
    # 2) DataEdit 未显式设置 duration，这里通过 monkey-patch 统一默认
    _patch_gr_toast_duration(args.toast_duration)

    # 便捷别名
    t = sf.g_language  # 文案多语言

    with gr.Blocks(title="LightSub WebUI") as demo:
        # 顶层 Tabs 容器（便于编程切换）
        with gr.Tabs() as tabs:
            # SubFix Tab（切分与修正）
            with gr.TabItem("切分与修正", id="subfix"):
                with gr.Row():
                    interval_slider = gr.Slider(minimum=0, maximum=2, value=0, step=0.01, label=t("Interval"), scale=3)
                    splitpoint_slider = gr.Slider(minimum=0, maximum=20.0, value=0, step=0.1, label=t("Audio Split Point(s)"), scale=3)

                    with gr.Column(min_width=150):
                        btn_merge_audio = gr.Button(t("Merge Audio"))
                        btn_audio_split = gr.Button(t("Split Audio"))

                    with gr.Column(min_width=150):
                        # 替换为“编辑文件” -> 跳转编辑Tab
                        btn_go_edit = gr.Button("编辑文件", variant="secondary")

                # 动态批量区域
                sf_text_list = []
                sf_audio_list = []
                sf_checkbox_list = []
                delete_btns = []
                with gr.Row():
                    with gr.Column():
                        for i in range(sf.g_batch):
                            with gr.Row():
                                text = gr.Textbox(label=t("Text"), visible=True, scale=5)
                                audio_output = gr.Audio(label=t("Output Audio"), visible=True, show_download_button=False, scale=5)
                                with gr.Column(min_width=120):
                                    audio_check = gr.Checkbox(label=t("Choose"))
                                    del_btn = gr.Button(t("Delete Audio"))
                                sf_text_list.append(text)
                                sf_audio_list.append(audio_output)
                                sf_checkbox_list.append(audio_check)
                                delete_btns.append(del_btn)

                with gr.Row():
                    index_slider = gr.Slider(minimum=0, maximum=sf.g_max_json_index, value=sf.g_index, step=1, label=t("Index"), scale=1)
                    with gr.Column(scale=1):
                        with gr.Row():
                            btn_previous_index = gr.Button(t("Previous Index"), scale=1)
                            btn_next_index = gr.Button(t("Next Index"))
                        btn_change_index = gr.Button(t("Change Index"))

            # 编辑 Tab（列表编辑）
            with gr.TabItem("编辑", id="edit") as tab_edit:
                gr.Markdown("## CSV 数据编辑器（wav_path,text）")
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

                df = gr.Dataframe(
                    headers=de.HEADERS,
                    value=[],
                    row_count=(0, "dynamic"),
                    col_count=de.N_COLS,
                    interactive=True,
                    label="未加载",
                )

                with gr.Row():
                    # 保存按钮需要在保存后同步刷新 SubFix 视图
                    btn_save = gr.Button("保存", variant="primary", scale=1)

        # 交互函数区域（放在 Blocks 内定义，便于闭包捕获组件）

        def go_edit_and_reload():
            # 切换到“编辑”Tab并刷新表格（从磁盘重新加载）
            try:
                de.init_from_args(sf.g_load_file)
            except Exception:
                pass
            return gr.update(selected="edit"), de._dataframe_update()

        btn_go_edit.click(go_edit_and_reload, inputs=[], outputs=[tabs, df])

        # SubFix 事件绑定
        btn_change_index.click(
            sf.b_change_index,
            inputs=[index_slider],
            outputs=[*sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        btn_previous_index.click(
            sf.b_previous_index,
            inputs=[],
            outputs=[index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        btn_next_index.click(
            sf.b_next_index,
            inputs=[],
            outputs=[index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        btn_merge_audio.click(
            sf.b_merge_audio,
            inputs=[interval_slider, *sf_checkbox_list],
            outputs=[index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        btn_audio_split.click(
            sf.b_audio_split,
            inputs=[splitpoint_slider, *sf_checkbox_list],
            outputs=[index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        # 删除按钮与文本动态保存
        for i, del_btn in enumerate(delete_btns):
            del_btn.click(
                sf.make_delete_row(i),
                inputs=[],
                outputs=[index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
            )

        for i, text_box in enumerate(sf_text_list):
            text_box.change(sf.make_submit_change_one(i), inputs=[text_box], outputs=[])

        # 编辑 Tab 交互
        btn_filter.click(de.ui_filter, inputs=[df, tb_query], outputs=[df])
        btn_reset_filter.click(de.ui_reset_filter, inputs=[], outputs=[df])
        btn_replace.click(de.ui_replace, inputs=[df, tb_find, tb_repl], outputs=[df])

        def ui_clear_replace_fields():
            return gr.update(value=""), gr.update(value=""), gr.update(value="")

        btn_clear_replace.click(ui_clear_replace_fields, inputs=[], outputs=[tb_query, tb_find, tb_repl])

        # 保存并刷新两端：
        # 1) 写回 CSV（使用 dataEdit 的保存逻辑）
        # 2) 刷新 DataEdit 的表格（_dataframe_update）
        # 3) 触发 SubFix 重新加载文件并刷新当前页面
        def save_and_refresh(df_value):
            de.ui_save(df_value)
            # DataEdit 刷新当前视图
            df_update = de._dataframe_update()
            # SubFix 重新加载并返回更新
            idx_update, *page_updates = sf.b_reload_file()
            return df_update, idx_update, *page_updates

        btn_save.click(
            save_and_refresh,
            inputs=[df],
            outputs=[df, index_slider, *sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )

        # 点击“编辑”Tab标题时自动刷新表格（从磁盘重新加载）
        def on_edit_tab_select():
            try:
                de.init_from_args(sf.g_load_file)
            except Exception:
                pass
            return de._dataframe_update()

        tab_edit.select(on_edit_tab_select, inputs=[], outputs=[df])

        # 页面加载时，初始化两个 Tab 的视图
        demo.load(
            sf.b_change_index,
            inputs=[gr.State(sf.g_index)],  # 使用当前索引初始化
            outputs=[*sf_text_list, *sf_audio_list, *sf_checkbox_list],
        )
        demo.load(de._dataframe_update, inputs=[], outputs=[df])

    return demo


def main():
    parser = argparse.ArgumentParser(description="LightSub WebUI（SubFix + 编辑）")
    parser.add_argument('--load_file', required=True, help='加载的 CSV 列表文件路径（包含表头: wav_path,text）')
    parser.add_argument('--g_batch', type=int, default=8, help='每页显示的音频数量, 默认: 8')
    parser.add_argument('--webui_language', default="zh", type=str, help='界面语言: zh 或 en, 默认: zh')
    parser.add_argument('--server_port', type=int, default=7860, help='WebUI端口, 默认: 7860')
    parser.add_argument('--toast_duration', type=float, default=3.0, help='全局提示条显示时长(秒), 默认: 3.0')
    args = parser.parse_args()

    demo = build_app(args)
    demo.launch(server_port=args.server_port)


if __name__ == '__main__':
    main()
