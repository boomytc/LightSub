import argparse
import csv
import os
import sys


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + ".csv" if ext else input_path + ".csv"


def convert_txt_to_csv(input_file: str, output_file: str) -> int:
    """
    将 list.txt 格式（每行：wav_path|text）转换为两列 CSV（无表头）。
    返回成功写入的行数。
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return 1

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for line in fin:
            total += 1
            s = line.strip()
            if not s:
                skipped += 1
                continue
            # 仅按第一次出现的竖线分割
            parts = s.split("|", 1)
            if len(parts) != 2:
                skipped += 1
                print(f"警告: 第{total}行未找到分隔符'|'，已跳过。", file=sys.stderr)
                continue
            wav_path, text = parts[0].strip(), parts[1].strip()
            writer.writerow([wav_path, text])
            written += 1

    print(f"完成: 总行数={total}, 写入={written}, 跳过={skipped}")
    print(f"输出文件: {output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="将 list.txt（wav_path|text）转换为两列 CSV（无表头）")
    parser.add_argument("input", nargs="?", default="dataset/audio_list/list.txt", help="输入的 txt 列表文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出 CSV 文件路径（默认与输入同名，后缀改为 .csv）")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output or default_output_path(input_file)
    return convert_txt_to_csv(input_file, output_file)


if __name__ == "__main__":
    raise SystemExit(main())

