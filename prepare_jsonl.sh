#!/bin/bash

# 显示帮助信息的函数
show_help() {
    echo "用途: 处理单个数据文件并生成用于语音识别训练的jsonl文件"
    echo
    echo "用法: $0 --input_file <输入文件> --output_dir <输出目录>"
    echo "或使用: $0 -h 显示帮助信息"
    echo
    echo "参数说明:"
    echo "  --input_file   输入文件路径，格式为 音频路径|讲话人ID|语言|文本内容"
    echo "  --output_dir   生成的文件的保存目录"
    echo "  --train_ratio  训练集比例 (默认: 0.8)"
    echo "  --val_ratio    验证集比例 (默认: 0.1)"
    echo
    echo "示例:"
    echo "  $0 --input_file ./data/input.txt --output_dir ./output"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --input_file)
            input_file="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --train_ratio)
            train_ratio="$2"
            shift 2
            ;;
        --val_ratio)
            val_ratio="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$input_file" ] || [ -z "$output_dir" ]; then
    echo "错误: 必须指定 --input_file 和 --output_dir"
    show_help
fi

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "错误: 输入文件 '$input_file' 不存在"
    exit 1
fi

# 获取脚本目录
script_dir=$(dirname "$(realpath "$0")")

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 构建命令参数
command_args=(
    "--input_file" "$input_file"
    "--output_dir" "$output_dir"
)

# 添加可选参数
if [ ! -z "$train_ratio" ]; then
    command_args+=("--train_ratio" "$train_ratio")
fi

if [ ! -z "$val_ratio" ]; then
    command_args+=("--val_ratio" "$val_ratio")
fi

# 统计输入文件行数
total_lines=$(wc -l < "$input_file")
echo "输入文件 '$input_file' 包含 $total_lines 行记录"

# 调用Python脚本处理数据
echo "开始处理数据文件..."
python3 "${script_dir}/prepare_jsonl.py" "${command_args[@]}"
process_result=$?

if [ $process_result -eq 0 ]; then
    echo -e "\n处理完成!"
    echo "----------------------------------------"
    echo "输出文件："
    echo "- ${output_dir}/train.jsonl"
    echo "- ${output_dir}/val.jsonl"
    echo "- ${output_dir}/test.jsonl"
    echo "----------------------------------------"
else
    echo -e "\n错误: 数据处理失败!"
    exit 1
fi