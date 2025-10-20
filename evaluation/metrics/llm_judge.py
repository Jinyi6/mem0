import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI
import os


LOCAL_MEM0_PATH = os.getenv("LOCAL_MEM0_PATH")
if not LOCAL_MEM0_PATH:
    raise ValueError("环境变量 LOCAL_MEM0_PATH 未设置，请在 .env 文件中配置。")

if not os.path.exists(LOCAL_MEM0_PATH):
    raise ImportError(f"指定的本地 mem0 路径不存在: {LOCAL_MEM0_PATH}")

import sys
if LOCAL_MEM0_PATH not in sys.path:
    sys.path.insert(0, LOCAL_MEM0_PATH)

print("="*80)
print(f"✅ 成功将本地 mem0 库路径添加到环境中: {LOCAL_MEM0_PATH}")
print("="*80)

from mem0.memory.utils import extract_json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# LOCOMO version

# ACCURACY_PROMPT = """
# Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
#     (1) a question (posed by one user to another user), 
#     (2) a ’gold’ (ground truth) answer, 
#     (3) a generated answer
# which you will score as CORRECT/WRONG.

# The point of the question is to ask about something one user should know about the other user based on their prior conversations.
# The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
# Question: Do you remember what I got the last time I went to Hawaii?
# Gold answer: A shell necklace
# The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

# For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

# Now it's time for the real question:
# Question: {question}
# Gold answer: {gold_answer}
# Generated answer: {generated_answer}

# First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
# Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

# Just return the label CORRECT or WRONG in a json format with the key as "label".
# """

# refined version within 10.1
# ACCURACY_PROMPT = """
# Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
#     (1) a question (posed by one user to another user),
#     (2) a ’gold’ (ground truth) answer,
#     (3) a generated answer
# which you will score as CORRECT/WRONG.

# The point of the question is to ask about something one user should know about the other user based on their prior conversations.
# The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
# Question: Do you remember what I got the last time I went to Hawaii?
# Gold answer: A shell necklace

# DECISION RULES (apply all):
# 1. EXACT FACT / CONTENT MATCH
#   - CORRECT only if the generated answer is semantically equivalent to the gold answer and does NOT contradict it.
#   - Ignore casing, punctuation, articles, and order when order is not semantically meaningful. Your judgement should be based on content itself, format is unrelative to its correctness. Synonyms/aliases are allowed only when they do not change the factual scope (e.g., “US” ≡ “United States”; “ten years” ≡ “10 years”; “Malia and Sasha” ≡ “Malia Obama and Sasha Obama”).
# 2. COVERAGE / SUBSET
#   - If the gold answer contains multiple required elements (entities, dates, attributes, list items, or facets), ALL must be present.
#   - If the generated answer provides only a subset of gold ⇒ WRONG (response ⊂ gold).
# Example: gold = “hiking, riding, running”; response = “hiking”.
#   - If the generated answer includes extra elements not in gold ⇒ WRONG (gold ⊂ response), unless those extras are purely paraphrastic/trivial and introduce no new facts.
# Example: gold = “agencies”; response = “agencies and lawyers” ⇒ WRONG.
# 3. ABSTRACTION LEVEL
#   - If the gold is ABSTRACT and the generated answer is MORE SPECIFIC (adds a specific date/entity/category not given), mark WRONG unless the added specificity is verbatim or entailed by the question (then allow).
#   - If the gold is SPECIFIC and the generated answer is MORE GENERAL (e.g., “console” for “Nintendo game console”,  "Ferrari" ≠ "classic vintage cars"), mark WRONG.
#   - Terminology equivalence (name - explanation/alias): When either the gold or the response is a specific term/proper name, treat the other as CORRECT if it gives a standard, identifying explanation or alias of the same term, without adding new facts or changing scope.
# 4. TEMPORAL CONSISTENCY:
#   - Dates/times must refer to the SAME calendar point or period as gold.
#   - Ranges must match; a single date is wrong if gold is "the Sunday before 25 May 2023"; a different year/month/day is wrong; the response providing a more specific date than gold answer is wrong (eg. While the golden answer is "2023", the response is "7 May 2023").
#   - Relative expressions ("last Tuesday") are CORRECT only if they unambiguously resolve to the SAME date/period as the gold answer. If ambiguous, mark WRONG.
# 5. NO-INFORMATION / CONTRADICTION:
#   - If the generated answer claims "not mentioned"/"no info"  ⇒ WRONG.
#   - Any factual contradiction with the gold (including YES/NO inversions, different amounts, different entities, different locations) ⇒ WRONG.
# 6. LINGUISTIC VARIATION:
#   - Minor wording changes, plural/singular, and true name variants are fine (e.g., "Malia and Sasha" for "Malia Obama and Sasha Obama").
    
# TIE-BREAK RULE:
# - If after applying the rules there is any uncertainty about equivalence or coverage, choose WRONG (favor precision over recall for reward stability).

# Now it's time for the real question:
# Question: {question}
# Gold answer: {gold_answer}
# Generated answer: {generated_answer}

# First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
# Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

# Just return the label CORRECT or WRONG in a json format with the key as "label":

# ```json
# {{
#     "label": "CORRECT" or "WRONG"
# }}
# """

# refined version within 10.12

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

Decision Rules (minimal, to reduce false negatives without raising false positives):

1) Entity & polarity lock (strict)
   - The generated answer must refer to the same thing/person/time as the gold; any different entity/time or negation flip => WRONG.
   - Examples:
     • Gold: “A shell necklace”; Gen: “A pearl necklace” => WRONG.
     • Gold: “He moved to Boston”; Gen: “He didn’t move to Boston” => WRONG.
     • Gold: “June 2023”; Gen: “July 2023” => WRONG.

2) Synonyms/aliases/format variants (allowed)
   - Accept clear paraphrases, aliases, and format variants that mean the same thing.
   - Also accept trivial form variations that keep the same concept: singular/plural/lemma changes; gerund vs noun phrasing; obvious short typos/truncations that still point to the same word.
   - Accept causal paraphrases that clearly keep the same cause (e.g., “advised/recommended/introduced/told to try” ≈ “advice”).
   - Examples:
     • Gold: “snakes”; Gen: “snake” => CORRECT.
     • Gold: “exploring the great outdoors”; Gen: “outdoor/outdoors/outdoor activities” => CORRECT.
     • Gold: “friend’s advice”; Gen: “a friend introduced/recommended it” => CORRECT.
     • Gold: “practicing basketball outside for hours”; Gen: “basketbal” (obvious cutoff) => CORRECT.
     • Gold: “03/15/2021”; Gen: “March 15, 2021” => CORRECT.

3) Specificity/entailment (one-way, head-preserving)
   - More specific answers that logically entail the gold are CORRECT.
   - Slight generalization is also CORRECT when it preserves the same head concept/topic and only drops modifiers or rolls up ONE level to the immediate parent category, without introducing alternatives.
   - If the gold includes essential qualifiers whose removal changes identity (e.g., flavor/type/brand among many siblings) OR the question is a reason (“why…?”) but the answer gives only a preference/statement (not a cause), => WRONG.
   - Examples (ACCEPT):
     • Gold: “work”; Gen: “work stress” => CORRECT (same head).
     • Gold: “cakes”; Gen: “baked goods” => CORRECT (immediate parent only).
     • Gold: “malfunctioning navigation app on the new phone”; Gen: “phone app malfunction” => CORRECT (immediate parent; same domain).
   - Examples (REJECT):
     • Gold: “pepperoni pizza”; Gen: “pizza/food/Italian food” => WRONG (too broad; many siblings).
     • Gold: “Coke”; Gen: “soda” => WRONG.
     • Gold (why): “Possibly because he likes to drink beer on his days off.”; Gen: “He prefers beer / prefers beer over Starbucks.” => WRONG (not a cause; missing condition).

4) Extra non-conflicting detail is fine; alternatives/hedging are not
   - Extra descriptive detail is allowed if it doesn’t contradict the gold. Listing alternatives or hedging counts as WRONG.
   - Examples:
     • Gold: “A shell necklace”; Gen: “A shell necklace from the street market” => CORRECT.
     • Gold: “Paris”; Gen: “Paris, France” => CORRECT.
     • Gold: “A shell necklace”; Gen: “A shell necklace or earrings” => WRONG.
     • Gold: “Paris”; Gen: “Maybe Paris?” => WRONG.

5) Lists, numbers, units, and dates (tight equivalence)
   - If the gold explicitly lists multiple required items, ALL must be present—no more, no less. Subsets/supersets/alternatives (“or/and-or/slash”) => WRONG.
   - Treat text as a list only when the gold clearly enumerates parallel items (and/&/slash “/”/the Chinese “、/和/与/以及”/comma-separated parallel nouns). Not a list when a comma is part of a single named item (e.g., “Washington, D.C.”) or a numeric separator (“1,000”).
   - Accept exact unit/date format equivalents (no value change). Do NOT accept vague approximations if the gold is exact.
   - For subjective descriptions, if the gold lists multiple concrete attributes (e.g., taste/texture/colour: “super good, rich and creamy”), the generated answer must preserve those attributes (or clear paraphrases). Generic sentiment alone (“loved it”, “good”) => WRONG.
     • Examples: Gold “Super good, rich and creamy”; Gen “He loved it” => WRONG; Gen “rich and creamy” => CORRECT.
   - Approximate quantifiers vs exact numbers are NOT equivalent (e.g., “a few/several/a couple” ≠ “4/3/2” unless the gold itself says that mapping). 
     • Examples: Gold “A few months”; Gen “4 months” => WRONG.
   - Date granularity must MATCH (no broadening or narrowing): Day ≠ Month; Month ≠ Year.
     • Examples: Gold “February, 2023”; Gen “8 February 2023” => WRONG.  Gold “March 15, 2021”; Gen “March 2021” => WRONG.
   - Additional list examples:
     • Gold: “different fantasy novels, characters, themes, and book recommendations”; Gen: “fantasy and book articles” => WRONG (missing listed items).
     • Gold: “cat and dog”; Gen: “cat” => WRONG; Gen: “cat and dog” => CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
"""

import time
import random
def evaluate_llm_judge(question, gold_answer, generated_answer):
    max_retries = 10
    retries = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-14B",
                messages=[
                    {
                        "role": "user",
                        "content": ACCURACY_PROMPT.format(
                            question=question, gold_answer=gold_answer, generated_answer=generated_answer
                        ),
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            label = json.loads(extract_json(response.choices[0].message.content))["label"]
            return 1 if label == "CORRECT" else 0

        except Exception as e:
            retries += 1
            error_message = str(e).lower() 
            
            print(f"⚠️  An error occurred: {str(e)}") 
            
            if retries >= max_retries:
                print(f"❌ Failed after max retries for question: {question}")
                return 0 

            if "tpm" in error_message or "limit" in error_message:
                sleep_duration = random.randint(40, 80)
                print(f"   -> Rate limit error detected. Waiting for {sleep_duration}s... (Attempt {retries}/{max_retries})")
                time.sleep(sleep_duration)
            else:
                sleep_duration = random.uniform(0.5, 1)  # 减少其他错误等待时间
                print(f"   -> Other error detected. Retrying in {sleep_duration:.1f}s... (Attempt {retries}/{max_retries})")
                time.sleep(sleep_duration)
    
    print(f"❌ Exhausted all retries for question: {question}")
    return 0



def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
