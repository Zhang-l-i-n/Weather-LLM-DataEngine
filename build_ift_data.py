import os
import json
import random
import argparse
import sys
from datetime import date, datetime, timedelta

def parse_arguments():
    parser = argparse.ArgumentParser(description="构建指令微调(IFT)数据集 (JSON格式)")
    parser.add_argument("--csv_dir", type=str, default="./forecast_csv", help="输入数据(CSV)所在目录")
    parser.add_argument("--report_dir", type=str, default="./report_by_llm", help="模型生成结果(报告+思考)所在目录")
    parser.add_argument("--instruction_file", type=str, default="./prompt/instruction.txt", help="存放指令的文件路径")
    parser.add_argument("--output_file", type=str, default="finetune_data.json", help="最终生成的 JSON 文件路径")
    
    # 日期参数
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2021-01-03", help="结束日期 (YYYY-MM-DD)")
    
    return parser.parse_args()

def generate_timestamps(start_date, end_date):
    """生成时间戳列表 YYYY-MM-DD_HHMMSS"""
    current_date = start_date
    date_format = "%Y-%m-%d"
    time_format = "%H%M%S"
    hours = [5, 11, 17, 20]
    formatted_timestamps = []

    while current_date <= end_date:
        for hour in hours:
            dt_object = datetime(current_date.year, current_date.month, current_date.day, hour)
            formatted_timestamps.append(f"{dt_object.strftime(date_format)}_{dt_object.strftime(time_format)}")
        current_date += timedelta(days=1)
    return formatted_timestamps

def load_instruction(file_path):
    """读取指令文件，如果不存在则使用默认指令"""
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 指令文件 {file_path} 不存在，使用默认通用指令。")
        return "You are a professional meteorologist. Analyze the provided weather data and generate a forecast report."
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    args = parse_arguments()
    
    # 1. 验证目录
    if not os.path.exists(args.csv_dir) or not os.path.exists(args.report_dir):
        print(f"❌ 错误: 输入目录不存在。\nCSV: {args.csv_dir}\nReport: {args.report_dir}")
        sys.exit(1)

    # 2. 准备数据
    try:
        s_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        e_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        print("❌ 错误: 日期格式需为 YYYY-MM-DD")
        sys.exit(1)

    instruction_text = load_instruction(args.instruction_file)
    timestamps = generate_timestamps(s_date, e_date)
    
    data_ift = []
    success_count = 0
    missing_count = 0

    print(f"--- 开始构建数据集 ({args.start_date} ~ {args.end_date}) ---")

    for ts in timestamps:
        csv_path = os.path.join(args.csv_dir, f"{ts}.csv")
        report_path = os.path.join(args.report_dir, f"{ts}.txt")
        think_path = os.path.join(args.report_dir, f"{ts}_think.txt")

        if not os.path.exists(csv_path) or not os.path.exists(report_path):
            missing_count += 1
            continue

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                input_data = f.read()

            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = f.read()

            final_output = ""
            if os.path.exists(think_path):
                with open(think_path, 'r', encoding='utf-8') as f:
                    think_content = f.read().strip()
                    
                if "<think>" in think_content:
                    think_content = think_content.replace('<__THINK__>', '<think>').replace('</__THINK__>', '</think>')
                    final_output = f"{think_content}\n\n{report_data}"
                else:
                    final_output = f"<think>\n{think_content}\n</think>\n\n{report_data}"
            else:
                final_output = report_data

            data_ift.append({
                "instruction": instruction_text,
                "input": input_data,
                "output": final_output
            })
            success_count += 1

        except Exception as e:
            print(f"处理 {ts} 时出错: {e}")

    random.shuffle(data_ift)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data_ift, f, ensure_ascii=False, indent=2)

    print(f"--- 构建完成 ---")
    print(f"✅ 成功条目: {success_count}")
    print(f"⏭️ 跳过条目: {missing_count} (文件缺失)")
    print(f"💾 保存至: {os.path.abspath(args.output_file)}")

if __name__ == '__main__':
    main()