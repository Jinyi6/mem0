#!/usr/bin/env python3
"""
分析两个评估结果JSON文件，比较它们的性能差异
"""

import json
import pandas as pd
from typing import Dict, List, Tuple, Any

def load_json_data(file_path: str) -> Dict[str, List[Dict]]:
    """加载JSON评估数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_question_results(data: Dict[str, List[Dict]]) -> List[Dict]:
    """从JSON数据中提取所有问题的结果"""
    all_results = []
    for user_id, results in data.items():
        for result in results:
            all_results.append({
                'user_id': user_id,
                'question': result['question'],
                'answer': result['answer'],
                'response': result['response'],
                'category': result['category'],
                'llm_score': result['llm_score']
            })
    return all_results

def find_matching_questions(results1: List[Dict], results2: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """找到两个结果集中匹配的问题"""
    matched_pairs = []
    
    # 创建问题到结果的映射
    questions1 = {r['question']: r for r in results1}
    questions2 = {r['question']: r for r in results2}
    
    # 找到共同的问题
    common_questions = set(questions1.keys()) & set(questions2.keys())
    
    for question in common_questions:
        matched_pairs.append((questions1[question], questions2[question]))
    
    return matched_pairs

def analyze_results(file1_path: str, file2_path: str):
    """分析两个评估文件的结果"""
    print("正在加载数据...")
    
    # 加载数据
    data1 = load_json_data(file1_path)
    data2 = load_json_data(file2_path)
    
    print(f"文件1包含 {len(data1)} 个用户的数据")
    print(f"文件2包含 {len(data2)} 个用户的数据")
    
    # 提取所有问题结果
    results1 = extract_question_results(data1)
    results2 = extract_question_results(data2)
    
    print(f"文件1包含 {len(results1)} 个问题结果")
    print(f"文件2包含 {len(results2)} 个问题结果")
    
    # 找到匹配的问题
    matched_pairs = find_matching_questions(results1, results2)
    print(f"找到 {len(matched_pairs)} 个匹配的问题")
    
    # 分析结果
    analysis_results = []
    
    for result1, result2 in matched_pairs:
        # 判断对错
        correct1 = result1['llm_score'] == 1
        correct2 = result2['llm_score'] == 1
        
        # 分类结果
        if correct1 and not correct2:
            category = "文件1对文件2错"
        elif not correct1 and correct2:
            category = "文件1错文件2对"
        elif correct1 and correct2:
            category = "两者都对"
        else:
            category = "两者都错"
        
        analysis_results.append({
            'question': result1['question'],
            'answer': result1['answer'],
            'response1': result1['response'][:200] + "..." if len(result1['response']) > 200 else result1['response'],
            'response2': result2['response'][:200] + "..." if len(result2['response']) > 200 else result2['response'],
            'score1': result1['llm_score'],
            'score2': result2['llm_score'],
            'category1': result1['category'],
            'category2': result2['category'],
            'correct1': correct1,
            'correct2': correct2,
            'comparison_category': category
        })
    
    # 创建DataFrame
    df = pd.DataFrame(analysis_results)
    
    # 统计各类别数量
    category_counts = df['comparison_category'].value_counts()
    print("\n=== 结果统计 ===")
    print(category_counts)
    
    # 保存详细结果到CSV
    output_file = '/Users/jinyi/Documents/code/memory/mem0/evaluation_comparison_results.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 显示一些示例
    print("\n=== 文件1对文件2错的示例 ===")
    file1_correct = df[df['comparison_category'] == '文件1对文件2错']
    if not file1_correct.empty:
        for idx, row in file1_correct.head(3).iterrows():
            print(f"\n问题: {row['question']}")
            print(f"答案: {row['answer']}")
            print(f"文件1得分: {row['score1']}")
            print(f"文件2得分: {row['score2']}")
            print(f"文件1回答: {row['response1']}")
            print(f"文件2回答: {row['response2']}")
    
    print("\n=== 文件1错文件2对的示例 ===")
    file2_correct = df[df['comparison_category'] == '文件1错文件2对']
    if not file2_correct.empty:
        for idx, row in file2_correct.head(3).iterrows():
            print(f"\n问题: {row['question']}")
            print(f"答案: {row['answer']}")
            print(f"文件1得分: {row['score1']}")
            print(f"文件2得分: {row['score2']}")
            print(f"文件1回答: {row['response1']}")
            print(f"文件2回答: {row['response2']}")
    
    return df

if __name__ == "__main__":
    # 文件路径
    file1_path = "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10/evaluation_metrics_judgerv2.json"
    file2_path = "/Users/jinyi/Documents/code/memory/mem0/evaluation/exp_data/locomo10_failed_6/locomo10_failed_6_top_k_30_filter_False_graph_False_3_1_0_20251013_131408/evaluation_metrics_20251014_184108.json"
    
    # 执行分析
    df = analyze_results(file1_path, file2_path)
