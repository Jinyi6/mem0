#!/usr/bin/env python3
"""
检查实验结果文件中是否包含记忆摘要和会话摘要记忆
"""
import json
import sys
import re

def is_session_summary_memory(memory_text):
    """
    判断一条记忆是否是会话摘要记忆（fact_abstract_mode生成）
    
    特征：
    1. 包含多个要点（以 - 开头）
    2. 包含换行符 \n
    3. 通常包含3个或更多要点
    """
    if ": " in memory_text:
        content = memory_text.split(": ", 1)[1]
    else:
        content = memory_text
    
    # 检查是否包含多个要点
    # 计算换行后的 - 数量
    bullet_count = len(re.findall(r'\n- |^- ', content))
    
    # 会话摘要通常包含3个或更多要点
    if bullet_count >= 3:
        return True
    
    # 或者检查是否包含换行符和至少2个要点
    if "\n" in content and content.count("- ") >= 2:
        return True
    
    return False

def is_long_term_profile_memory(memory_text):
    """
    判断一条记忆是否是长期画像记忆
    """
    return "[Long-term Profile]:" in memory_text

def check_memory_types(json_file_path):
    """
    检查JSON文件中的记忆类型
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = 0
    questions_with_summary = 0
    total_session_summaries = 0
    total_long_term_profiles = 0
    
    for conversation_id, questions in data.items():
        for qa_result in questions:
            total_questions += 1
            
            # 检查记忆摘要（summarize_memories）
            if "speaker_1_memory_summary" in qa_result or "speaker_2_memory_summary" in qa_result:
                questions_with_summary += 1
            
            # 检查会话摘要记忆和长期画像记忆
            speaker_1_memories = qa_result.get("speaker_1_memories", [])
            speaker_2_memories = qa_result.get("speaker_2_memories", [])
            
            for mem in speaker_1_memories + speaker_2_memories:
                if is_session_summary_memory(mem):
                    total_session_summaries += 1
                if is_long_term_profile_memory(mem):
                    total_long_term_profiles += 1
    
    print("=" * 80)
    print(f"文件: {json_file_path}")
    print("=" * 80)
    print(f"\n📊 统计结果：")
    print(f"  总问题数: {total_questions}")
    print(f"\n1️⃣  记忆摘要（summarize_memories）:")
    print(f"  包含记忆摘要的问题数: {questions_with_summary}")
    print(f"  覆盖率: {questions_with_summary/total_questions*100:.1f}%")
    
    print(f"\n2️⃣  会话摘要记忆（fact_abstract_mode）:")
    print(f"  在记忆列表中出现的会话摘要记忆数: {total_session_summaries}")
    print(f"  平均每个问题: {total_session_summaries/total_questions:.2f} 条")
    
    print(f"\n3️⃣  长期画像记忆（long_term_profile_mode）:")
    print(f"  在记忆列表中出现的长期画像记忆数: {total_long_term_profiles}")
    print(f"  平均每个问题: {total_long_term_profiles/total_questions:.2f} 条")
    
    print("\n" + "=" * 80)
    
    # 显示一些示例
    print("\n📝 示例：")
    print("\n会话摘要记忆示例（fact_abstract_mode）:")
    example_count = 0
    for conversation_id, questions in data.items():
        for qa_result in questions:
            speaker_1_memories = qa_result.get("speaker_1_memories", [])
            for mem in speaker_1_memories[:5]:  # 只检查前5条
                if is_session_summary_memory(mem):
                    print(f"  {mem[:150]}...")
                    example_count += 1
                    if example_count >= 2:
                        break
            if example_count >= 2:
                break
        if example_count >= 2:
            break
    
    print("\n长期画像记忆示例（long_term_profile_mode）:")
    example_count = 0
    for conversation_id, questions in data.items():
        for qa_result in questions:
            speaker_1_memories = qa_result.get("speaker_1_memories", [])
            for mem in speaker_1_memories:
                if is_long_term_profile_memory(mem):
                    print(f"  {mem[:150]}...")
                    example_count += 1
                    if example_count >= 1:
                        break
            if example_count >= 1:
                break
        if example_count >= 1:
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_memory_types.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    check_memory_types(json_file_path)

