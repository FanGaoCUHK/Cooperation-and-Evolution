import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import warnings
import os
import glob
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  


COOPERATE = 'C'
DEFECT = 'D'


def payoff(my_move, opp_move):
    if my_move == COOPERATE and opp_move == COOPERATE:
        return 2
    elif my_move == COOPERATE and opp_move == DEFECT:
        return -1
    elif my_move == DEFECT and opp_move == COOPERATE:
        return 3
    else:
        return 0

STRATEGY_FOLDER = r"E:\coin\AI\AI strategy-raw data"
ALL_AI_TEXTS = []   

def load_all_strategy_texts(folder_path):
    """读取文件夹下所有 run_*.txt 文件，返回文本列表"""
    texts = []
    pattern = os.path.join(folder_path, "run_*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到策略文件，请检查路径：{folder_path}")
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        texts.append(text)
    return texts

ALL_AI_TEXTS = load_all_strategy_texts(STRATEGY_FOLDER)

def parse_strategy_text(text):
    """
    从自然语言策略文本中提取规则，返回一个字典 rules，包含：
      - initial_move: 'C' 或 'D'
      - phases: 列表，每个元素为 (start, end, type, params)
         type 可以是 'always_C', 'always_D', 'TFT', 'forgiving_TFT' 等
      - special_forgive: (连续背叛次数阈值, 动作) 或 None
    解析逻辑针对常见的结构：第1轮单独给出；中间轮次基于TFT+偶尔原谅；终局全部背叛。
    """
    rules = {
        'initial_move': COOPERATE,   
        'phases': [],              
        'forgive_after_consecutive': None  
    }
    
    first_match = re.search(r'第\s*1\s*轮[：:]\s*选择\s*(合作|欺骗)', text)
    if first_match:
        decision_word = first_match.group(1)
        rules['initial_move'] = COOPERATE if '合作' in decision_word else DEFECT

    endgame_match = re.search(r'第\s*(\d+)\s*轮\s*至\s*第?\s*30\s*轮[：:]\s*无论对手历史行为如何，?\s*全部\s*(合作|欺骗)', text)
    if endgame_match:
        endgame_start = int(endgame_match.group(1))
        end_action = COOPERATE if '合作' in endgame_match.group(2) else DEFECT
        rules['phases'].append((endgame_start, 30, 'always', end_action))
    
    middle_match = re.search(r'第\s*(\d+)\s*轮\s*至\s*第?\s*(\d+)\s*轮[：:]', text)
    if middle_match:
        start = int(middle_match.group(1))
        end = int(middle_match.group(2))
        if re.search(r'对手上一轮选择合作.*本轮选择合作', text) and re.search(r'对手上一轮选择欺骗.*本轮选择欺骗', text):
            rule_type = 'TFT'
            forgive_match = re.search(r'连续欺骗了\s*(\d+)\s*轮.*下一轮.*合作', text)
            if forgive_match:
                consecutive_n = int(forgive_match.group(1))
                rules['forgive_after_consecutive'] = (consecutive_n, COOPERATE)
                rule_type = 'TFT_with_forgive'
            rules['phases'].append((start, end, rule_type, None))
        else:
            rules['phases'].append((start, end, 'TFT', None))
    
    if not rules['phases']:
        rules['phases'].append((1, 30, 'TFT', None))
    
    return rules

class Player:
    def __init__(self, strategy):
        self.strategy = strategy
        self.total_score = 0
        
        if self.strategy == 'AI':
            chosen_text = random.choice(ALL_AI_TEXTS)
            self.ai_rules = parse_strategy_text(chosen_text)
        self.reset_game()
    
    def reset_game(self):
        self.opponent_last_move = None
        self.betrayed_count = 0
        self.previous_move = None
        self.previous_payoff = None
        self.consecutive_opp_defects = 0
        self.just_forgave = False
        
    def make_move(self, round_num):
        if self.strategy == 'ALLC':
            return COOPERATE
        elif self.strategy == 'ALLD':
            return DEFECT
        elif self.strategy == 'TFT':
            if round_num == 1:
                return COOPERATE
            else:
                return self.opponent_last_move
        elif self.strategy == 'GTFT':
            if round_num == 1:
                return COOPERATE
            else:
                if self.opponent_last_move == COOPERATE:
                    return COOPERATE
                else:
                    if random.random() < 0.3:
                        return COOPERATE
                    else:
                        return DEFECT
        elif self.strategy == 'WSLS':
            if round_num == 1:
                return COOPERATE
            else:
                if self.previous_payoff >= 2:
                    return self.previous_move
                else:
                    if self.previous_move == COOPERATE:
                        return DEFECT
                    else:
                        return COOPERATE
        elif self.strategy == 'AI':
            rules = self.ai_rules
            if round_num == 1:
                return rules['initial_move']
            
            current_phase = None
            for start, end, rtype, extra in rules['phases']:
                if start <= round_num <= end:
                    current_phase = (start, end, rtype, extra)
                    break
            if current_phase is None:
                return self.opponent_last_move if self.opponent_last_move is not None else COOPERATE
            
            phase_start, phase_end, rtype, extra = current_phase
            
            if rtype == 'always':
                return extra  
            
            if rtype in ('TFT', 'TFT_with_forgive'):
                forgive_rule = rules.get('forgive_after_consecutive')
                if forgive_rule and not self.just_forgave:
                    consec_needed, forgive_action = forgive_rule
                    if self.consecutive_opp_defects >= consec_needed and self.opponent_last_move == DEFECT:
                        self.just_forgave = True
                        return forgive_action
                if self.opponent_last_move is not None:
                    return self.opponent_last_move
                else:
                    return COOPERATE  
            
            return COOPERATE  
                    
    def update_after_move(self, self_move, opp_move, round_num, pay):
        self.opponent_last_move = opp_move
        
        if self.strategy == 'AI':
            if opp_move == DEFECT:
                self.consecutive_opp_defects += 1
            else:
                self.consecutive_opp_defects = 0
                self.just_forgave = False   
        
        if self.strategy == 'AI' and opp_move == DEFECT:
            self.betrayed_count += 1
        if self.strategy == 'WSLS':
            self.previous_payoff = pay
            self.previous_move = self_move

num_simulations = 100
rounds_per_game = 30
num_generations = 8
num_eliminate = 200

human_strategies = ['TFT', 'GTFT', 'ALLC', 'ALLD', 'WSLS']
num_human_per_strategy = 200
num_ai_initial = 200

players = []
for strategy in human_strategies:
    for i in range(num_human_per_strategy):
        players.append(Player(strategy))
for i in range(num_ai_initial):
    players.append(Player('AI'))

generation_scores = []
generation_counts = []
anova_results = []
all_strategies = human_strategies + ['AI']

for gen in range(num_generations):
    for player in players:
        player.total_score = 0
    
    for sim in range(num_simulations):
        random.shuffle(players)
        for i in range(0, len(players), 2):
            p1 = players[i]
            p2 = players[i+1]
            p1.reset_game()
            p2.reset_game()
            for round_num in range(1, rounds_per_game + 1):
                move1 = p1.make_move(round_num)
                move2 = p2.make_move(round_num)
                pay1 = payoff(move1, move2)
                pay2 = payoff(move2, move1)
                p1.total_score += pay1
                p2.total_score += pay2
                p1.update_after_move(move1, move2, round_num, pay1)
                p2.update_after_move(move2, move1, round_num, pay2)
    
    scores_dict = {s: [] for s in all_strategies}
    counts_dict = {s: 0 for s in all_strategies}
    for player in players:
        if player.strategy in scores_dict:
            scores_dict[player.strategy].append(player.total_score)
            counts_dict[player.strategy] += 1
    
    avg_scores = {}
    for strategy, scores in scores_dict.items():
        if scores:
            avg_scores[strategy] = np.mean(scores)
        else:
            avg_scores[strategy] = 0
    
    generation_scores.append(avg_scores)
    generation_counts.append(counts_dict)
    
    valid_strategies = [s for s in all_strategies if len(scores_dict[s]) >= 2]
    if len(valid_strategies) >= 2:
        anova_data = [scores_dict[s] for s in valid_strategies]
        f_val, p_val = stats.f_oneway(*anova_data)
        anova_results.append((f_val, p_val, valid_strategies))
        print(f"第{gen+1}代完成:")
        for strategy in all_strategies:
            print(f"  {strategy}: 平均分 = {avg_scores[strategy]:.2f}, 数量 = {counts_dict[strategy]}")
        print(f"  ANOVA 结果: F = {f_val:.4f}, p = {p_val:.4e}")
        if p_val < 0.05:
            print("  ANOVA 显示至少有一个策略存在显著差异（p < 0.05）")
        else:
            print("  ANOVA 显示无显著差异")
        if p_val < 0.05 and len(valid_strategies) > 2:
            print("  事后检验 (Bonferroni校正):")
            pairs = list(combinations(valid_strategies, 2))
            num_tests = len(pairs)
            for s1, s2 in pairs:
                t, p = stats.ttest_ind(scores_dict[s1], scores_dict[s2])
                adjusted_p = min(1, p * num_tests)
                sig = "*" if adjusted_p < 0.05 else ""
                print(f"    {s1} vs {s2}: t = {t:.4f}, p = {p:.4e}, adjusted_p = {adjusted_p:.4e}{sig}")
    else:
        anova_results.append((None, None, valid_strategies))
        print(f"第{gen+1}代完成:")
        for strategy in all_strategies:
            print(f"  {strategy}: 平均分 = {avg_scores[strategy]:.2f}, 数量 = {counts_dict[strategy]}")
        print("  ANOVA: 数据不足，无法进行分析")
    
    if gen < num_generations - 1:
        players.sort(key=lambda x: x.total_score)
        players = players[num_eliminate:]
        candidate_strategies = [s for s in all_strategies if counts_dict[s] > 0]
        if not candidate_strategies:
            best_strategy = 'AI'
        else:
            best_strategy = max(candidate_strategies, key=lambda s: avg_scores[s])
        print(f"  最高分策略: {best_strategy} (平均分 = {avg_scores[best_strategy]:.2f})")
        for i in range(num_eliminate):
            players.append(Player(best_strategy))
        print(f"  淘汰并替换了{num_eliminate}个最低分玩家，添加了{num_eliminate}个{best_strategy}策略玩家")
    print()

fig, ax = plt.subplots(figsize=(8, 8))
generations = range(1, num_generations + 1)
for strategy in all_strategies:
    scores = [gen_scores[strategy] for gen_scores in generation_scores]
    ax.plot(generations, scores, marker='o', label=strategy)
ax.set_xlabel('Generation')
ax.set_ylabel('Mean Score')
ax.set_title('Evolution of Strategy Scores across Generations')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

elimination_dict = {}
for strategy in all_strategies:
    counts = [gen_counts[strategy] for gen_counts in generation_counts]
    for gen_idx, count in enumerate(counts):
        if count == 0:
            elimination_dict[strategy] = gen_idx + 1
            break

elimination_by_gen = {}
for strategy, gen in elimination_dict.items():
    if gen not in elimination_by_gen:
        elimination_by_gen[gen] = []
    elimination_by_gen[gen].append(strategy)

for gen, strategies in elimination_by_gen.items():
    if len(strategies) == 1:
        label = strategies[0]
    else:
        strategies_sorted = sorted(strategies)
        label = '; '.join(strategies_sorted)
    first_strategy = strategies[0]
    scores = [gen_scores[first_strategy] for gen_scores in generation_scores]
    score_at_elimination = scores[gen-1]
    ax.annotate(label, 
               xy=(gen, score_at_elimination), 
               xytext=(gen, score_at_elimination + 15),
               textcoords='data',
               ha='center',
               va='bottom',
               arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n最终代统计:")
final_scores = generation_scores[-1]
final_counts = generation_counts[-1]
for strategy in all_strategies:
    print(f"{strategy}: 平均分 = {final_scores[strategy]:.2f}, 数量 = {final_counts[strategy]}")