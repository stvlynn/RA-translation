import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

threads = 4 #线程数，默认4

# OpenAI API 配置
API_TOKEN = "set_your_token_here"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 默认的系统提示
DEFAULT_SYSTEM_PROMPT = "请将以下文本翻译成中文，但保留用“<>”包含的内容不变"

client = OpenAI(api_key=API_TOKEN, base_url=BASE_URL)

def translate_text(text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

def translate_chunk(chunk, output_file, pbar):
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for line in chunk:
            data = json.loads(line)
            data['prompt'] = translate_text(data['prompt']) #翻译的列
            data['response'] = translate_text(data['response']) #翻译的列
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            pbar.update(1)

def translate_jsonl_file(input_file, output_file, num_tasks=threads):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    chunk_size = len(lines) // num_tasks
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with tqdm(total=len(lines), desc="总翻译进度") as pbar:
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [executor.submit(translate_chunk, chunk, output_file, pbar) for chunk in chunks]
            for future in futures:
                future.result()

# 示例调用
translate_jsonl_file('input.jsonl', 'output.jsonl')