import pandas as pd
import openai
import time
import os

API_KEY = ""           
BASE_URL = ""
MODEL = ""              
EXCEL_PATH = r""
OUTPUT_DIR = "strategy_outputs"          
N_RUNS = 100                             
TEMPERATURE = 0.7         


GAME_RULES = """
## 信任硬币游戏规则
1. 每轮你和对手同时选择：
   - 合作(C): 放入一枚硬币，对方获得三枚硬币
   - 欺骗(D): 不放硬币

2. 收益规则：
   - 双方合作：各得+2（净收益）
   - 你合作，对手欺骗：你得-1，对手得+3
   - 你欺骗，对手合作：你得+3，对手得-1
   - 双方都欺骗：各得0

3. 当前游戏总轮数：30轮
"""

df = pd.read_excel(EXCEL_PATH)
history_text = ""
for match_id, match_df in df.groupby('Source.Name'):
    history_text += f"\n对局 {match_id}:\n"
    for _, row in match_df.iterrows():
        rnd = row['round']
        p1 = row['player1_decision']
        p2 = row['player2_decision']
        history_text += f"  第{rnd}轮: 玩家1选{p1}, 玩家2选{p2}\n"

base_prompt = f"""你是一位博弈玩家。以下是多次“信任硬币游戏”的历史记录。
{GAME_RULES}

历史对局记录
{history_text}

任务
请根据这些历史记录，设计一个能在这类博弈中表现优异的固定策略。
"""

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, N_RUNS + 1):
    print(f" {i}/{N_RUNS} ")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": base_prompt}
            ],
            temperature=TEMPERATURE,
            max_completion_tokens = 50000,
        )
        raw_output = response.choices[0].message.content
        clean_output = raw_output.strip()
        filename = os.path.join(OUTPUT_DIR, f"run_{i:03d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(clean_output)
        print(f"   {filename}")

    except Exception as e:
        print(f"   {i} 失败: {e}")
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a", encoding="utf-8") as err_f:
            err_f.write(f"Run {i}: {e}\n")

    time.sleep(1.5)

print(f"\n全部完成！输出文件位于文件夹：{OUTPUT_DIR}")