import argparse
import copy
import os
import uuid
import subprocess
import platform

import librosa
import csv
import json
import gradio as gr
import numpy as np
import soundfile

# 全局变量区域  
g_load_file = "" # 要加载的文件路径

g_max_json_index = 0
g_index = 0
g_batch = 8
g_text_list = []
g_audio_list = []
g_checkbox_list = []
g_data_json = []
g_language = None

# Toast 显示时长（秒）
TOAST_DURATION = 3


# 语言配置字典,支持中英文界面切换
SUBFIX_LANG_CONFIG_MAP = {
    "zh": {
        "Change Index" : "改变索引",
        "Merge Audio" : "合并音频",
        "Delete Audio" : "删除音频",
        "Previous Index" : "前一页",
        "Next Index" : "后一页",
        "Choose" : "选择",
        "Output Audio" : "音频播放",
        "Text" : "文本",
        "Split Audio" : "分割音频",
        "Audio Split Point(s)" : "音频分割点(单位：秒)",
        "Index":"索引",
        "Interval":"合并间隔（单位：秒）",
        "Open in Editor" : "在编辑器中打开",
        "Reload File" : "重新加载文件"
    },
}


# 文本国际化处理类
class SUBFIX_TextLanguage():
    """
    处理WebUI界面的多语言显示
    language: 语言代码，如'en'/'zh'
    """
    def __init__(self, language : str = "en") -> None:
        if language in SUBFIX_LANG_CONFIG_MAP.keys():
            self.language = language
        else:
            self.language = "en"
        pass

    def get_text(self, text : str) -> str:
        if self.language == "en":
            return text
        elif text in SUBFIX_LANG_CONFIG_MAP[self.language].keys() :
            return SUBFIX_LANG_CONFIG_MAP[self.language][text]
        else:
            return text
        
    def __call__(self, text : str) -> str:
        return self.get_text(text)


# 重新加载数据，根据index和batch大小返回数据切片
def reload_data(index, batch):
    """
    根据索引和批次大小重新加载数据
    index: 起始索引
    batch: 每批次加载数量
    return: 数据切片列表
    """
    global g_index
    g_index = index
    global g_batch
    g_batch = batch
    datas = g_data_json[index:index+batch]
    output = []
    for d in datas:
        output.append(
            {
                "text": d["text"],
                "wav_path": d["wav_path"]
            }
        )
    return output


# 更改当前显示的数据索引
def b_change_index(index):
    """
    改变当前显示的数据索引位置
    index: 新的索引位置
    return: 更新后的UI组件列表
    """
    global g_index, g_batch
    g_index = index
    try:
        save_index_json()
    except Exception as _:
        pass
    datas = reload_data(index, g_batch)
    output = []
    # 文本框：使用gr.update设置标签/值
    for i in range(g_batch):
        if i < len(datas):
            output.append(gr.update(label=f"{g_language('Text')} {i+index}", value=datas[i]["text"]))
        else:
            output.append(gr.update(label=g_language("Text"), value=""))
    # 音频组件：设置路径值或None
    for i in range(g_batch):
        output.append(datas[i]["wav_path"] if i < len(datas) else None)
    # 复选框：默认重置为False
    for _ in range(g_batch):
        output.append(False)
    return output


def b_next_index():
    """
    下一页：忽略传入的滑块值，基于当前页面真实索引翻页。
    - 无入参：使用全局 g_index/g_batch 翻页
    """
    global g_index, g_batch, g_max_json_index
    # 基于当前页面真实索引与批大小进行翻页
    start = g_index
    size = g_batch
    # 允许跳到最后一页（可能是非满页）
    new_index = min(start + size, max(g_max_json_index, 0))
    return gr.update(value=new_index), *b_change_index(new_index)


def b_previous_index():
    """
    上一页：忽略传入的滑块值，基于当前页面真实索引翻页。
    - 无入参：使用全局 g_index/g_batch 翻页
    """
    global g_index, g_batch
    start = g_index
    size = g_batch
    new_index = max(start - size, 0)
    return gr.update(value=new_index), *b_change_index(new_index)


def make_submit_change_one(local_idx: int):
    """
    返回一个仅更新当前行文本的回调，避免全量刷新。
    local_idx: 当前页面内的行索引 [0, g_batch)
    回调签名: fn(new_text) -> None (无输出)
    """
    def _fn(new_text: str):
        global g_data_json, g_index
        if new_text is None:
            return
        abs_idx = g_index + local_idx
        if 0 <= abs_idx < len(g_data_json):
            new_text_stripped = str(new_text).strip()
            if g_data_json[abs_idx]["text"].strip() != new_text_stripped:
                g_data_json[abs_idx]["text"] = new_text_stripped
                b_save_file()
        # 无输出，避免触发组件重绘
        return
    return _fn


def make_delete_row(local_idx: int):
    """
    返回一个仅删除当前行的回调。
    local_idx: 当前页面内的行索引 [0, g_batch)
    回调签名: fn() -> (index_slider_update, *page_updates)
    """
    def _fn():
        global g_data_json, g_index, g_max_json_index
        abs_idx = g_index + local_idx
        if 0 <= abs_idx < len(g_data_json):
            path = g_data_json[abs_idx]["wav_path"]
            try:
                print("删除文件", path)
                os.remove(path)
            except Exception as e:
                print("删除失败:", e)
            g_data_json.pop(abs_idx)
            g_max_json_index = len(g_data_json) - 1
            if g_index > g_max_json_index:
                g_index = g_max_json_index if g_max_json_index >= 0 else 0
            # 删除后始终保存列表变更
            b_save_file()
        return gr.update(value=g_index, maximum=(g_max_json_index if g_max_json_index>=0 else 0)), *b_change_index(g_index)
    return _fn


def get_next_path(filename):
    """
    生成下一个可用的文件路径
    filename: 原始文件名
    return: 可用的新文件路径
    """
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path) :
            return new_path
    return os.path.join(base_dir, f'{str(uuid.uuid4())}.wav')


# 音频分割功能
def b_audio_split(audio_breakpoint, *checkbox_list):
    """
    将选中的音频文件按时间点分割成两段
    audio_breakpoint: 分割时间点(秒)
    checkbox_list: 选中的音频文件复选框状态
    return: 更新后的UI状态
    """
    global g_data_json , g_max_json_index
    checked_index = []
    for i, checkbox in enumerate(checkbox_list):
        if (checkbox == True and g_index+i < len(g_data_json)):
            checked_index.append(g_index + i)
    if len(checked_index) == 1 :
        index = checked_index[0]
        audio_json = copy.deepcopy(g_data_json[index])
        path = audio_json["wav_path"]
        data, sample_rate = librosa.load(path, sr=None, mono=True)
        audio_maxframe = len(data)
        break_frame = int(audio_breakpoint * sample_rate)

        if (break_frame >= 1 and break_frame < audio_maxframe):
            audio_first = data[0:break_frame]
            audio_second = data[break_frame:]
            nextpath = get_next_path(path)
            soundfile.write(nextpath, audio_second, sample_rate)
            soundfile.write(path, audio_first, sample_rate)
            g_data_json.insert(index + 1, audio_json)
            g_data_json[index + 1]["wav_path"] = nextpath
            b_save_file()

    g_max_json_index = len(g_data_json) - 1
    return gr.update(value=g_index, maximum=g_max_json_index), *b_change_index(g_index)
    
# 合并音频功能    
def b_merge_audio(interval_r, *checkbox_list):
    """
    将多个选中的音频文件合并为一个
    interval_r: 音频间隔时间(秒)
    checkbox_list: 选中的音频文件复选框状态
    return: 更新后的UI状态
    """
    global g_data_json , g_max_json_index
    checked_index = []
    audios_path = []
    audios_text = []
    delete_files = []
    for i, checkbox in enumerate(checkbox_list):
        if (checkbox == True and g_index+i < len(g_data_json)):
            checked_index.append(g_index + i)
            
    if (len(checked_index)>1):
        for i in checked_index:
            audios_path.append(g_data_json[i]["wav_path"])
            audios_text.append(g_data_json[i]["text"])
        for i in reversed(checked_index[1:]):
            delete_files.append(g_data_json[i]["wav_path"])
            g_data_json.pop(i)

        base_index = checked_index[0]
        base_path = audios_path[0]
        g_data_json[base_index]["text"] = "".join(audios_text)

        audio_list = []
        l_sample_rate = None
        for i, path in enumerate(audios_path):
            data, sample_rate = librosa.load(path, sr=l_sample_rate, mono=True)
            l_sample_rate = sample_rate
            if (i > 0):
                silence = np.zeros(int(l_sample_rate * interval_r))
                audio_list.append(silence)

            audio_list.append(data)

        audio_concat = np.concatenate(audio_list)

        for item_file in delete_files:
            os.remove(item_file)

        soundfile.write(base_path, audio_concat, l_sample_rate)

        b_save_file()
    
    g_max_json_index = len(g_data_json) - 1
    
    return gr.update(value=g_index, maximum=g_max_json_index), *b_change_index(g_index)


# 保存文件
def b_save_file():
    """保存数据到文件：根据扩展名选择写入 txt 或 csv"""
    ext = os.path.splitext(g_load_file)[1].lower()
    if ext == '.csv':
        with open(g_load_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 写表头，标准两列：wav_path, text
            writer.writerow(["wav_path", "text"])
            for data in g_data_json:
                writer.writerow([data.get("wav_path", ""), data.get("text", "")])
    else:
        with open(g_load_file, 'w', encoding='utf-8') as file:
            for data in g_data_json:
                wav_path = data.get("wav_path", "")
                text = data.get("text", "")
                # 使用简化格式：仅保存音频路径和文本内容
                file.write(f"{wav_path}|{text}".strip()+'\n')


# 加载文件
def b_load_file():
    """从文件加载数据：支持 txt(list) 与 csv 两种格式"""
    global g_data_json, g_max_json_index
    g_data_json = []
    ext = os.path.splitext(g_load_file)[1].lower()
    
    if ext == '.csv':
        with open(g_load_file, 'r', encoding='utf-8', newline='') as source:
            reader = csv.reader(source)
            first = True
            for row in reader:
                if not row:
                    continue
                # 兼容可能存在的表头
                if first and len(row) >= 2 and row[0].strip().lower() == 'wav_path' and row[1].strip().lower() == 'text':
                    first = False
                    continue
                first = False
                wav_path = (row[0] if len(row) >= 1 else '').strip()
                text = (row[1] if len(row) >= 2 else '').strip()
                if not wav_path:
                    continue
                audio_id = os.path.splitext(os.path.basename(wav_path))[0]
                g_data_json.append({
                    'wav_path': wav_path,
                    'speaker_name': audio_id,
                    'language': 'AUTO',
                    'text': text
                })
    else:
        with open(g_load_file, 'r', encoding='utf-8') as source:
            for line in source:
                s = line.strip()
                if not s:
                    continue
                parts = s.split('|', 1)
                if len(parts) == 2:
                    wav_path, text = parts[0], parts[1]
                    audio_id = os.path.splitext(os.path.basename(wav_path))[0]
                    g_data_json.append({
                        'wav_path': wav_path,
                        'speaker_name': audio_id,
                        'language': 'AUTO',
                        'text': text.strip()
                    })
                else:
                    print(f"错误行格式 (期望2个字段): {parts}")

    g_max_json_index = len(g_data_json) - 1


def get_index_json_path():
    try:
        base_dir = os.path.dirname(g_load_file)
        return os.path.join(base_dir if base_dir else '.', 'index.json')
    except Exception:
        return 'index.json'


def save_index_json():
    path = get_index_json_path()
    state = {"index": int(max(0, g_index))}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False)


def load_index_json():
    global g_index
    path = get_index_json_path()
    try:
        if not os.path.exists(path):
            # 不存在则创建默认索引文件
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({"index": 0}, f, ensure_ascii=False)
            g_index = 0
            return
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        idx = int(state.get('index', 0))
        if g_max_json_index is not None and g_max_json_index >= 0:
            idx = max(0, min(idx, g_max_json_index))
        else:
            idx = 0
        g_index = idx
    except Exception as e:
        print(f"加载索引状态失败: {e}")


def b_open_in_editor():
    """在默认编辑器中打开当前文件（使用 toast 通知反馈）"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", g_load_file])
        elif platform.system() == "Windows":
            subprocess.run(["start", g_load_file], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", g_load_file])
        # 成功提示（使用全局时长）
        gr.Info(f"文件已在编辑器中打开: {g_load_file}", duration=TOAST_DURATION)
        return
    except Exception as e:
        # 失败提示（使用全局时长）
        gr.Error(f"打开编辑器失败: {e}", duration=TOAST_DURATION)
        return


def b_reload_file():
    """重新加载文件并刷新界面（使用 toast 通知反馈）"""
    global g_index, g_max_json_index
    try:
        # 保存当前索引位置
        current_index = g_index
        
        # 重新加载文件
        b_load_file()
        
        # 确保索引不越界
        if current_index > g_max_json_index:
            current_index = max(0, g_max_json_index)
        
        g_index = current_index
        
        # 返回更新后的界面状态
        index_update = gr.update(value=current_index, maximum=(g_max_json_index if g_max_json_index >= 0 else 0))
        page_updates = b_change_index(current_index)

        # 成功提示（使用全局时长）
        gr.Info(f"文件重新加载成功，共 {len(g_data_json)} 条数据", duration=TOAST_DURATION)
        return index_update, *page_updates
        
    except Exception as e:
        # 失败提示（使用全局时长），同时保持界面稳定
        gr.Error(f"重新加载文件失败: {e}", duration=TOAST_DURATION)
        safe_index_update = gr.update(value=g_index, maximum=(g_max_json_index if g_max_json_index >= 0 else 0))
        return safe_index_update, *b_change_index(g_index)


def set_global(load_file, batch, webui_language):
    """
    设置全局变量
    load_file: 要加载的文件路径
    batch: 批次大小
    webui_language: 界面语言
    """
    global g_load_file, g_batch, g_language

    g_batch = int(batch)
    g_load_file = load_file if load_file != "None" else "dataset/audio_list/list.csv"
    g_language = SUBFIX_TextLanguage(webui_language)

    b_load_file()
    # 加载上次浏览进度（索引），目录级 index.json
    load_index_json()


# WebUI主函数
def subfix_startwebui(args):
    """
    启动WebUI服务
    args: 命令行参数,包含:
        - load_file: 加载的文件路径
        - g_batch: 每页显示数量
        - webui_language: 界面语言
        - server_port: 服务端口
    """
    set_global(args.load_file, args.g_batch, args.webui_language)
    
    with gr.Blocks() as demo:
            
        with gr.Row():
            interval_slider = gr.Slider(
                    minimum=0, maximum=2, value=0, step=0.01, label=g_language("Interval"), scale=3
            )
            splitpoint_slider = gr.Slider(
                    minimum=0, maximum=20.0, value=0, step=0.1, label=g_language("Audio Split Point(s)"), scale=3
            )
            
            with gr.Column(min_width=150):
                btn_merge_audio = gr.Button(g_language("Merge Audio"))
                btn_audio_split = gr.Button(g_language("Split Audio"))

            with gr.Column(min_width=150):
                btn_open_editor = gr.Button(g_language("Open in Editor"), variant="secondary")
                btn_reload_file = gr.Button(g_language("Reload File"), variant="secondary")

        with gr.Row():
            with gr.Column():
                delete_btns = []
                for i in range(0,g_batch):
                    with gr.Row():
                        text = gr.Textbox(
                            label = g_language("Text"),
                            visible = True,              
                            scale=5
                        )
                        audio_output = gr.Audio(
                            label= g_language("Output Audio"),
                            visible = True,
                            show_download_button = False,
                            scale=5,
                        )
                        with gr.Column(min_width=120):
                            audio_check = gr.Checkbox(
                                label=g_language("Choose"),
                            )
                            del_btn = gr.Button(g_language("Delete Audio"))
                        g_text_list.append(text)
                        g_audio_list.append(audio_output)
                        g_checkbox_list.append(audio_check)
                        delete_btns.append(del_btn)

        with gr.Row():
            index_slider = gr.Slider(
                    minimum=0, maximum=g_max_json_index, value=g_index, step=1, label=g_language("Index"), scale=1
            )
            with gr.Column(scale=1):
                with gr.Row():
                    btn_previous_index = gr.Button(g_language("Previous Index"), scale=1)
                    btn_next_index = gr.Button(g_language("Next Index"))

                btn_change_index = gr.Button(g_language("Change Index"))
                
        btn_change_index.click(
            b_change_index,
            inputs=[
                index_slider,
            ],
            outputs=[
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ],
        )
        
        btn_previous_index.click(
            b_previous_index,
            inputs=[],
            outputs=[
                index_slider,
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ],
        )
        
        btn_next_index.click(
            b_next_index,
            inputs=[],
            outputs=[
                index_slider,
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ],
        )

        btn_merge_audio.click(
            b_merge_audio,
            inputs=[
                interval_slider,
                *g_checkbox_list
            ],
            outputs=[
                index_slider,
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ]
        )

        btn_audio_split.click(
            b_audio_split,
            inputs=[
                splitpoint_slider,
                *g_checkbox_list
            ],
            outputs=[
                index_slider,
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ]
        )

        btn_open_editor.click(
            b_open_in_editor,
            inputs=[],
            outputs=[]
        )

        btn_reload_file.click(
            b_reload_file,
            inputs=[],
            outputs=[
                index_slider,
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ]
        )

        # 将删除按钮绑定在列表完全填充后，确保输出覆盖所有行
        for i, del_btn in enumerate(delete_btns):
            del_btn.click(
                make_delete_row(i),
                inputs=[],
                outputs=[
                    index_slider,
                    *g_text_list,
                    *g_audio_list,
                    *g_checkbox_list
                ],
            )

        # 为文本框变化添加自动保存功能（每行最小更新）
        for i, text_box in enumerate(g_text_list):
            text_box.change(
                make_submit_change_one(i),
                inputs=[text_box],
                outputs=[],
            )

        demo.load(
            b_change_index,
            inputs=[
                index_slider,
            ],
            outputs=[
                *g_text_list,
                *g_audio_list,
                *g_checkbox_list
            ],
        )
        
    demo.launch(server_port = args.server_port)


if __name__ == "__main__":
    parser_subfix_webui = argparse.ArgumentParser(description='SubFix WebUI - 专用于datasets_list_create.py生成的数据文件')
    parser_subfix_webui.add_argument('--load_file', required=True, help='加载的列表文件路径，推荐 CSV (wav_path,text，带表头); 兼容旧版 txt')
    parser_subfix_webui.add_argument('--g_batch', default=8, help='每页显示的音频数量, 默认: 8')
    parser_subfix_webui.add_argument('--webui_language', default="zh", type=str, help='界面语言: zh 或 en, 默认: zh')
    parser_subfix_webui.add_argument('--server_port', default=7860, type=int, help='WebUI端口, 默认: 7860')

    parser_subfix = parser_subfix_webui.parse_args()

    subfix_startwebui(parser_subfix)
