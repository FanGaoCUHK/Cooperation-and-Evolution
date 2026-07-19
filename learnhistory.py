import os
import pandas as pd
from openai import OpenAI
import json


os.environ["OPENAI_API_KEY"] = "your api key"  # use your own api key
client = OpenAI()


OVERALL_RECORD_PATH = r""# use your own location of historical data to save and update for ai agent

def get_overall_record():
    """读取总体游戏记录并计算每一轮的合作率"""
    try:
        if os.path.exists(OVERALL_RECORD_PATH):
            df = pd.read_excel(OVERALL_RECORD_PATH)
            if not df.empty:
                
                round_coop_rates = {}
                for round_num in range(1, 31):
                    round_data = df[df['round'] == round_num]
                    if not round_data.empty:
                        p1_coop_rate = round_data[round_data['player1_decision'] == 'C'].shape[0] / round_data.shape[0]
                        p2_coop_rate = round_data[round_data['player2_decision'] == 'C'].shape[0] / round_data.shape[0]
                        round_coop_rates[round_num] = (p1_coop_rate, p2_coop_rate)
                
                
                record_str = "历史游戏每一轮合作率统计:\n"
                record_str += "轮次 | 玩家1合作率 | 玩家2合作率\n"
                record_str += "-----|------------|------------\n"
                
                for round_num, (p1_rate, p2_rate) in round_coop_rates.items():
                    record_str += f"{round_num:3}  | {p1_rate*100:8.1f}%   | {p2_rate*100:8.1f}%\n"
                
                
                total_games = df['Source.Name'].nunique()
                record_str += f"\n基于 {total_games} 场历史游戏的统计"
                return record_str
        return "无历史游戏记录"
    except Exception as e:
        print(f"读取总体记录时出错: {e}")
        return "无历史游戏记录"


def payoff(p1, p2):
    if p1 == "C" and p2 == "C":
        return 2, 2
    elif p1 == "C" and p2 == "D":
        return -1, 3
    elif p1 == "D" and p2 == "C":
        return 3, -1
    else:
        return 0, 0


PROMPT_TEMPLATE = """
你是{player}，一个博弈玩家。你正在参与信任硬币博弈，尝试获胜。

重要提示：在做出决策前，请先仔细学习以下历史游戏记录的合作率统计，分析对手的典型行为模式：

{coop_rate_str}

游戏规则：
- 选择 "C"（合作）则自己付出1，对方获得+3
- 选择 "D"（欺骗）则自己不付出，对方得0

收益规则：
- 双方合作：各 +2
- 自己合作对方欺骗：自己 -1，对方 +3
- 自己欺骗对方合作：自己 +3，对方 -1
- 双方欺骗：各 0

当前轮次: {round_num}/30
当前游戏的历史对局:
{history}

请基于历史合作率分析和当前游戏情况，做出最优决策。
问题：请你只回答一个字母 "C" 或 "D"，不要附加任何解释。
"""

def ask_model(player, round_num, history, coop_rate_str):
    prompt = PROMPT_TEMPLATE.format(
        player=player,
        round_num=round_num,
        coop_rate_str=coop_rate_str,
        history=json.dumps(history, ensure_ascii=False, indent=2),
    )
    
    response = client.chat.completions.create(
        model="your model",#you should make sure your account have enough right and money to use the model you set here
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip().upper()


all_results = []


coop_rate_str = get_overall_record()
print("历史合作率数据加载完成：")
print(coop_rate_str)

for game_id in range(1, 101):  # You can set the number of automatic runs by yourself.
    print(f"\n=== 开始第 {game_id} 场游戏 ===")
    
    
    print("AI正在学习历史合作率数据...")
    
    history = []
    total_score_p1, total_score_p2 = 0, 0

    for round_num in range(1, 31):  
        choice_p1 = ask_model("Player1", round_num, history, coop_rate_str)
        if choice_p1 not in ["C", "D"]:
            choice_p1 = "C"

        choice_p2 = ask_model("Player2", round_num, history, coop_rate_str)
        if choice_p2 not in ["C", "D"]:
            choice_p2 = "C"

        gain_p1, gain_p2 = payoff(choice_p1, choice_p2)
        total_score_p1 += gain_p1
        total_score_p2 += gain_p2

        round_result = {
            "game_id": game_id,
            "round": round_num,
            "player1_choice": choice_p1,
            "player2_choice": choice_p2,
            "gain_p1": gain_p1,
            "gain_p2": gain_p2,
            "total_p1": total_score_p1,
            "total_p2": total_score_p2,
        }
        history.append(round_result)
        print(f"[Game {game_id}] Round {round_num}: P1={choice_p1}, P2={choice_p2}, Score=({gain_p1},{gain_p2})")

    
    filename = f"trust_game_result_{game_id}.xlsx"
    df_history = pd.DataFrame(history)
    df_history.to_excel(filename, index=False)

    print(f"第 {game_id} 场游戏结束！结果已保存到 {filename}")
    print(f"最终得分: Player1={total_score_p1}, Player2={total_score_p2}")

    all_results.append({
        "game_id": game_id,
        "final_score_p1": total_score_p1,
        "final_score_p2": total_score_p2,
        "history": history
    })


summary_filename = "trust_game_summary.xlsx"
summary_data = []

for result in all_results:
    
    game_history = result["history"]
    p1_coop_count = sum(1 for round_data in game_history if round_data["player1_choice"] == "C")
    p2_coop_count = sum(1 for round_data in game_history if round_data["player2_choice"] == "C")
    
    summary_data.append({
        "game_id": result["game_id"],
        "final_score_p1": result["final_score_p1"],
        "final_score_p2": result["final_score_p2"],
        "player1_avg_score": result["final_score_p1"] / 30,
        "player2_avg_score": result["final_score_p2"] / 30,
        "player1_coop_rate": p1_coop_count / 30,
        "player2_coop_rate": p2_coop_count / 30
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_excel(summary_filename, index=False)

print(f"\n所有 10 场游戏结束！")
print(f"详细结果已保存到 trust_game_result_1.xlsx 到 trust_game_result_10.xlsx")
print(f"汇总结果已保存到 {summary_filename}")

print("\n=== 实验总体统计 ===")
avg_p1_score = df_summary['final_score_p1'].mean()
avg_p2_score = df_summary['final_score_p2'].mean()
avg_p1_coop = df_summary['player1_coop_rate'].mean()
avg_p2_coop = df_summary['player2_coop_rate'].mean()

print(f"Player1 平均最终得分: {avg_p1_score:.2f}")
print(f"Player2 平均最终得分: {avg_p2_score:.2f}")
print(f"Player1 平均合作率: {avg_p1_coop:.1%}")
print(f"Player2 平均合作率: {avg_p2_coop:.1%}")


all_games_data = []
for result in all_results:
    all_games_data.extend(result["history"])

df_all_games = pd.DataFrame(all_games_data)
df_all_games.to_excel("trust_game_all_resultslearning2nd.xlsx", index=False)
print("所有游戏的详细数据已保存到 trust_game_all_resultslearning2nd.xlsx")
