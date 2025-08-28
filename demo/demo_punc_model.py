from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model="models/punc_models/punc_ct", 
        disable_update=True
    )

# text = '我们都是木头人不会讲话不会动'
text = '增 速 阶 段 的 注 意 力 分 配 以 外 为 主 余 光 兼 顾 座 舱'

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    result = model.generate(input=text)

print(f"输入文本: {text}")
print(f"预测结果: {result}")
print(f"处理后的文本: {result[0]['text']}")

# 如果想处理文件中的文本
# with open('example/punc_example.txt', 'r', encoding='utf-8') as f:
#     text_from_file = f.read().strip()
#     result_from_file = model.generate(input=text_from_file)
#     print(f"文件文本: {text_from_file}")
#     print(f"文件预测结果: {result_from_file}")