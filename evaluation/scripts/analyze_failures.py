
import json

file_path = "evaluation/exp_data/long/evaluation_metrics_differences_summary_20251208_150442.json"

MODE_0_ID = "20251204-035221_1"
MODE_1410_ID = "20251207-070239_1"

with open(file_path, "r") as f:
    data = json.load(f)

print(f"Total keys: {list(data.keys())}")
questions = data.get("questions", [])
print(f"Total questions: {len(questions)}")

count = 0
for item in questions:
    correct_list = item.get("correct", [])
    wrong_list = item.get("wrong", [])
    
    if MODE_0_ID in correct_list and MODE_1410_ID in wrong_list:
        count += 1
        print("-" * 50)
        print(f"Question: {item.get('question', '').strip()}")
        print(f"Ground Truth: {item.get('answer', '')}")
        
        methods = item.get("methods", {})
        
        # Mode 0 Data
        m0_data = methods.get(MODE_0_ID, {})
        m0_resp = m0_data.get("response", "N/A").strip()
        print(f"\n[Mode 0 (Correct)] Response: {m0_resp}")
        
        # Mode 14.10 Data
        m14_data = methods.get(MODE_1410_ID, {})
        m14_resp = m14_data.get("response", "N/A").strip()
        print(f"\n[Mode 14.10 (Wrong)] Response: {m14_resp}")
        
        # Briefly check memories count
        m0_mem_count = len(m0_data.get("speaker_1_memories", [])) + len(m0_data.get("speaker_2_memories", []))
        m14_mem_count = len(m14_data.get("speaker_1_memories", [])) + len(m14_data.get("speaker_2_memories", []))
        print(f"\nMemories Retrieved: Mode 0 = {m0_mem_count}, Mode 14.10 = {m14_mem_count}")
        
        # Optional: Print first few memories of 14.10 to see if noise is obvious
        params = m14_data.get("speaker_1_memories", [])[:3]
        if params:
            print(f"Sample 14.10 Memories: {params}")

print(f"\nFound {count} cases where Mode 0 was correct and Mode 14.10 was wrong.")
