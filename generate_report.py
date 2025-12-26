from datetime import date, datetime, timedelta
import os
import json
import argparse
import sys
from typing import Dict, Any, List
import pandas as pd
import tqdm
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ========== 配置环境变量 ==========
load_dotenv() 

# 预先检查环境变量
api_key = os.getenv("CHAT_API_KEY") 
base_url = os.getenv("CHAT_API_BASE_URL")
model_name = os.getenv("CHAT_MODEL", "Qwen3-32B")

if not api_key:
    raise ValueError("错误: 未在 .env 文件中找到 CHAT_API_KEY")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="基于 CSV 数据调用大模型生成天气报告")
    
    parser.add_argument(
        "--start_date", 
        type=str, 
        default="2021-01-01", 
        help="开始日期，格式 YYYY-MM-DD (默认: 2021-01-01)"
    )
    parser.add_argument(
        "--end_date", 
        type=str, 
        default="2021-01-03", 
        help="结束日期，格式 YYYY-MM-DD (默认: 2021-01-03)"
    )
    parser.add_argument(
        "--csv_dir", 
        type=str, 
        default="./forecast_csv", 
        help="CSV 文件所在的目录路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./report_by_llm", 
        help="生成的报告保存路径"
    )
    
    return parser.parse_args()

def validate_date(date_str):
    """
    验证并转换日期字符串
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"❌ 错误: 日期格式 '{date_str}' 无效，请使用 YYYY-MM-DD 格式")
        sys.exit(1)

def generate_timestamps(start_date: date, end_date: date):
    """
    生成时间戳列表，格式: YYYY-MM-DD_HHMMSS
    """
    current_date = start_date
    date_format = "%Y-%m-%d"
    time_format = "%H%M%S"
    hours = [5, 11, 17, 20]
    formatted_timestamps = []

    while current_date <= end_date:
        for hour in hours:
            dt_object = datetime(current_date.year, current_date.month, current_date.day, hour)
            date_str = dt_object.strftime(date_format)
            time_str = dt_object.strftime(time_format)
            formatted_timestamps.append(f"{date_str}_{time_str}")
        current_date += timedelta(days=1)

    return formatted_timestamps

def extract_think_and_content(text: str) -> List[str]:
    """
    解析模型输出，分离思考过程(<think>)和最终结果。
    """
    if '<think>' in text and '</think>' in text:
        parts = text.split('</think>', 1)
        think_part = parts[0].replace('<think>', '').strip()
        report_part = parts[1].strip()
        return [think_part, report_part]
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        return [parts[0].strip(), parts[1].strip()]
    else:
        return ["", text.strip()]

def get_single_response(chat_model, user_prompt, max_retries=2):
    """
    调用大模型并处理重试逻辑
    """
    for attempt in range(max_retries + 1):
        try:
            response = chat_model.invoke([{"role": "user", "content": user_prompt}])
            content = response.content
            if content:
                return extract_think_and_content(content)
        except Exception as e:
            print(f"  [Attempt {attempt+1}] API 调用失败: {e}")
            if attempt == max_retries:
                return ["", ""]
    return ["", ""]

def main():
    # 1. 解析参数
    args = parse_arguments()
    start_date = validate_date(args.start_date)
    end_date = validate_date(args.end_date)
    csv_dir = args.csv_dir
    report_dir = args.output_dir

    if start_date > end_date:
        print("❌ 错误: 开始日期不能晚于结束日期")
        return

    print(f"--- 启动批量天气报告生成 ---")
    print(f"📅 日期范围: {start_date} 至 {end_date}")
    print(f"📂 输入目录: {csv_dir}")
    print(f"📂 输出目录: {report_dir}")
    
    # 2. 准备目录
    os.makedirs(report_dir, exist_ok=True)
    
    # 3. 初始化模型
    chat_model = ChatOpenAI(
        model=model_name, 
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,
        max_tokens=8192,
        stop=["<|im_end|>"]
    )

    # 4. 读取 Prompt 模板
    prompt_path = './prompt/forecast.txt'
    if not os.path.exists(prompt_path):
        print(f"❌ 错误: 找不到 Prompt 文件 {prompt_path}")
        return

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 5. 生成待处理的时间列表
    datelist = generate_timestamps(start_date=start_date, end_date=end_date)
    print(f"📝 预计处理 {len(datelist)} 个时次的数据")
    
    # 6. 主循环
    success_count = 0
    for day_str in tqdm.tqdm(datelist):
        try:
            csv_path = os.path.join(csv_dir, f'{day_str}.csv')
            
            if not os.path.exists(csv_path):
                print(f"  ⚠️ 警告: {csv_path} 不存在")
                continue

            with open(csv_path, 'r', encoding='utf-8') as f:
                raw_csv_content = f.read()
            
            if not raw_csv_content:
                continue 
    
            user_prompt_final = prompt_template.replace('<!INPUT!>', raw_csv_content)
            [llm_think, final_report] = get_single_response(chat_model, user_prompt_final)
            
            if not final_report:
                print(f"  ⚠️ 警告: {day_str} 生成内容为空")
                continue

            # 保存结果
            with open(os.path.join(report_dir, f"{day_str}.txt"), "w", encoding="utf-8") as f:
                f.write(final_report)
            
            if llm_think:
                with open(os.path.join(report_dir, f"{day_str}_think.txt"), "w", encoding="utf-8") as f:
                    f.write(llm_think)
            
            success_count += 1

        except Exception as e:
            print(f"❌ {day_str} 处理发生异常: {e}")
                    
    print(f"--- 任务结束: 成功生成 {success_count}/{len(datelist)} 份报告 ---")

if __name__ == '__main__':
    main()