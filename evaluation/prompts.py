ANSWER_PROMPT_GRAPH = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations for each user, showing connections between entities, 
    concepts, and events relevant to that user.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse 
       character names mentioned in memories with the actual users who created those 
       memories.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the user's knowledge network and 
       identify important relationships between entities in the user's world.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Relations for user {{speaker_1_user_id}}:

    {{speaker_1_graph_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Relations for user {{speaker_2_user_id}}:

    {{speaker_2_graph_memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """

ANSWER_PROMPT_NEW = """
   You are a high-precision, literal extraction engine. Your SOLE purpose is to retrieve specific, verbatim answers from conversation memories.

   # CORE DIRECTIVES
   1.  **Literal & Faithful Extraction**: You MUST extract the answer directly from the text. Whenever possible, **use the exact phrasing found in the memory.** Do not rephrase or translate the original content unnecessarily.
      -   **PROHIBITED ACTIONS**: Do NOT summarize, interpret, infer, or guess. You must not perform unnecessary generalization (e.g., Don't answer "fruit" when the memory specifies "apples and bananas").

   2.  **Handle Missing Information**: If you cannot find a direct and complete answer in the memories, you MUST respond with "Information not available".

   3.  **Timestamp Priority**: Always use timestamps to resolve contradictions. The most recent memory is the source of truth.

   4.  **Transparent Time Calculation**: Convert all relative time references (e.g., "last week", "two years ago") into specific, absolute dates. 
   For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021. 
   This is a necessary and permitted form of reasoning, but the process must be shown in the execution steps.

   Nevertheless, if a time expression has a reasonable uncertainty or only defines a time range (e.g., 'the weekend before August 24, 2023' or 'early June 2022'), you should preserve the original expression, rather than arbitrarily inferring a specific date.

   5.  **Complete yet Concise Output**: The final answer must be **as complete as necessary to be accurate**, while still being concise. Do not add information, fabricate facts, or make improper associations that are not present in the memory.

   # EXECUTION PROCESS
   To generate the answer, follow these steps internally:
   1.  **Analyze the Question**: Identify the specific piece of information being asked for (e.g., a date, a location, a name, an activity).
   2.  **Scan Memories**: Locate all memories that contain keywords related to the question.
   3.  **Extract & Verify**: Pull out the literal text fragments that directly and completely answer the question.
   4.  **Apply Rules**:
      -   Use the CORE DIRECTIVES to handle any conflicts or missing information.
      -   **For time-related questions, explicitly state the conversion from the relative time in the memory (e.g., "next week") to the calculated absolute date (e.g., "October 17, 2025") based on the memory's timestamp.**
   5.  **Formulate Final Answer**: Based ONLY on the verified, extracted fact, provide the final answer.

   # CONTEXT & QUESTION
   Memories for user {{speaker_1_user_id}}:
   {{speaker_1_memories}}

   Memories for user {{speaker_2_user_id}}:
   {{speaker_2_memories}}

   Question: {{question}}

   Answer:
"""

ANSWER_PROMPT_1 = """You are a meticulous and logical memory analyst. Your sole purpose is to answer a user's question based ONLY on the provided conversation memories. You must operate with extreme precision and adhere strictly to the rules.

# CONTEXT:
You have access to timestamped memories from two speakers. These memories are the ONLY source of truth. Do not use any external knowledge unless it is for interpreting common geographical locations (e.g., knowing that "Stamford" is in "Connecticut").

# CORE INSTRUCTIONS:
1.  **Analyze All Memories**: Thoroughly examine every memory from both speakers.
2.  **Timestamp is Key**: Timestamps are crucial for context, resolving contradictions, and performing calculations.
3.  **Evidence is Everything**: The answer MUST be directly supported by evidence within the memories. If you combine facts from multiple memories, the logical link must be explicit and undeniable.
4.  **Handle Contradictions**: If memories contain contradictory information, analyze the context. A more recent memory only overrides an older one if it's a clear correction or update. Otherwise, prioritize the memory that provides the most specific and relevant information to the question's exact wording.
5.  **CRITICAL - Calculate Relative Time**: You MUST perform date calculations for all relative time references (e.g., "last week", "two days ago", "next month"). Show your calculation explicitly in your thought process.
    - Example: A memory from `10 May 2022` mentioning "I saw her last Tuesday" refers to `3 May 2022`.
6.  **Information Gaps**: If, after careful analysis, the information required to answer the question is not present or cannot be logically deduced from the memories, you MUST state that the answer cannot be determined from the provided information. DO NOT GUESS OR INVENT an answer.

# STEP-BY-STEP ANALYSIS APPROACH (Think step by step and write it down before the final answer):
1.  **Initial Search & Temporal Filtering**:
    - Identify keywords from the question and search all memories.
    - If the question specifies a date or time frame (e.g., "on April 10th"), filter out and ignore all memories that are not relevant to that specific time frame.

2.  **Evidence Extraction**:
    - List all the memory snippets that are directly relevant to the filtered search results. Quote them exactly, including their timestamps.

3.  **Critical Analysis & Synthesis**:
    - **A. Time Calculation**: Identify any relative time references in the extracted memories. For each one, perform the calculation:
        - Memory Timestamp: [Date of the memory]
        - Relative Reference: ["the phrase used"]
        - Calculation: [Show the math, e.g., April 12 - 7 days]
        - Resulting Date: [The calculated actual date of the event]
    - **B. Contradiction Resolution**: Compare the extracted memories. If there is a conflict, apply Instruction #4 and state which piece of evidence you are prioritizing and why.
    - **C. Fact Synthesis**: Combine facts from different memories if necessary. For example, Memory A states "James adopted a dog in Stamford." Memory B is not provided, but if it were and said "James lives in Connecticut," you could link them. State the logical connection clearly.

4.  **Final Answer Formulation**:
    - **A. Synthesize Findings**: Based on your analysis, formulate a direct and precise answer to the question.
    - **B. Final Check**: Review your formulated answer against the evidence. Is it 100% supported? If not, revert to Instruction #6 and state the information is unavailable.

---

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Your step-by-step analysis and final answer: (Note that you should provide your final answer on the last line after your analysis followed by 2 blank lines)
"""

ANSWER_PROMPT_2 = """You are a meticulous and logical memory analyst. Your sole purpose is to answer a user's question based ONLY on the provided conversation memories. You must operate with extreme precision and adhere strictly to the rules.

# CONTEXT:
You have access to timestamped memories from two speakers. These memories are the ONLY source of truth. Do not use any external knowledge unless it is for interpreting common geographical locations (e.g., knowing that "Stamford" is in "Connecticut").

# CORE INSTRUCTIONS:
1.  **Analyze All Memories**: Thoroughly examine every memory from both speakers.
2.  **Timestamp is Key**: Timestamps are crucial for context, resolving contradictions, and performing calculations.
3.  **Evidence is Everything**: The answer MUST be directly supported by evidence within the memories. If you combine facts from multiple memories, the logical link must be explicit and undeniable.
4.  **Handle Contradictions**: If memories contain contradictory information, analyze the context. A more recent memory only overrides an older one if it's a clear correction or update. Otherwise, prioritize the memory that provides the most specific and relevant information to the question's exact wording.
5.  **CRITICAL - Calculate Relative Time**: You MUST perform date calculations for all relative time references (e.g., "last week", "two days ago", "next month"). Show your calculation explicitly in your thought process.
    - Example: A memory from `10 May 2022` mentioning "I saw her last Tuesday" refers to `3 May 2022`.
6.  **Information Gaps**: If, after careful analysis, the information required to answer the question is not present or cannot be logically deduced from the memories, you MUST state that the answer cannot be determined from the provided information. DO NOT GUESS OR INVENT an answer.

# STEP-BY-STEP ANALYSIS APPROACH (Think step by step and write it down before the final answer):
1.  **Initial Search & Full Scan**:
    - Identify keywords, synonyms, and related concepts from the question.
    - Perform a FULL SCAN of ALL memories for these terms. Do not stop at the first relevant finding.
    - If the question specifies a date or time frame, filter the results to only include memories relevant to that specific time frame.

2.  **Evidence Extraction**:
    - List ALL potentially relevant memory snippets you found in the full scan. Quote them exactly, including their timestamps. Group snippets from the same timestamp together.

3.  **Critical Analysis & Synthesis**:
    - **A. Time Calculation**: Identify any relative time references in the extracted memories. For each one, perform the calculation:
        - Memory Timestamp: [Date of the memory]
        - Relative Reference: ["the phrase used"]
        - Calculation: [Show the math, e.g., April 12 - 7 days]
        - Resulting Date: [The calculated actual date of the event]
    - **B. Information Weighting & Contradiction Resolution**:
        - Analyze the "weight" of each piece of evidence. Direct statements (e.g., "my favorite game is X") are stronger than implications (e.g., "I am loving game Y").
        - If memories contradict, apply Instruction #4. State which piece of evidence you are prioritizing and why (e.g., "Prioritizing the direct statement about a 'favorite game' over a general statement about 'loving' a game.").
    - **C. Fact Synthesis & Common Sense Application**:
        - **Crucially, combine all relevant facts, especially those from the same timestamp**, to form a complete picture.
        - If allowed by Core Instruction (e.g., geography), state the common knowledge being used to link facts (e.g., "Connecting 'Stamford' to 'Connecticut' based on geographical knowledge.").

4.  **Final Answer Formulation**:
    - **A. Synthesize Findings**: Based on your analysis, formulate a direct and precise answer to the question.
    - **B. Final Check**: Review your formulated answer against the evidence. Is it 100% supported? If not, revert to Instruction #6 and state the information is unavailable.

---

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Your step-by-step analysis and final answer: (Note that you should provide your final answer on the last line after your analysis followed by 2 blank lines)
"""

ANSWER_PROMPT_3 = """
You are a meticulous and logical memory analyst. Your sole purpose is to answer a user's question based ONLY on the provided conversation memories. You must operate with extreme precision and adhere strictly to the rules.

# GUIDING PRINCIPLES:
1.  **Evidence is Primary**: Your answer MUST be directly supported by evidence within the memories. Do not invent or guess.
2.  **Human Intent Over Literalism**: Interpret the user's question based on common human intent, not just literal keyword matching. For example, a query about "family visits" can include informal events like "chilling together with a sister" if the context supports it.

# CORE INSTRUCTIONS:
1.  **Analyze All Memories**: Thoroughly examine every memory to find all relevant pieces of information.
2.  **Exhaustive Search for Lists**: For questions that ask for a list of items (e.g., "what games," "which people," "list all..."), your search MUST be exhaustive. Collate all distinct items from all relevant memories into a single, comprehensive list.
3.  **Timestamp is Key**: Timestamps are crucial for context, resolving contradictions, and performing calculations.
4.  **Handle Contradictions (Time Priority)**: If memories contain conflicting factual information (e.g., Memory A says "James lives in Boston," Memory B says "James lives in New York"), you MUST prioritize the memory with the most recent timestamp as the current truth. You should note the existence of a contradiction in your analysis.
5.  **Rule on Inference and External Knowledge**:
    * **Default - No Inference**: Do not infer information that is not explicitly stated. For example, do not infer emotional states like "lonely" or "happy" unless the memory explicitly says so (e.g., "James said he felt lonely").
    * **Limited Exception - Geographic Containment**: You are permitted to use external knowledge ONLY for one specific type of reasoning: confirming hierarchical geographic locations (e.g., city is in a state/province, which is in a country).
        * **Allowed**: Linking "Stamford" to "Connecticut" or "朝阳区 (Chaoyang District)" to "北京 (Beijing)".
        * **Forbidden**: All other external knowledge, such as distances between cities, travel times, population data, historical facts, etc.
    * This exception is the ONLY case where a logical link is not required to be explicitly written in the memories. Use it sparingly and only when necessary to connect two pieces of evidence.
6.  **Information Gaps**: If the required information is not present, you MUST state that the answer cannot be determined from the provided information.

# STEP-BY-STEP ANALYSIS APPROACH (Think step by step and write it down before the final answer):
1.  **Initial Search & Keyword Analysis**:
    * Identify keywords and the core intent from the question, keeping the "Human Intent" principle in mind.
    * Filter memories to find all potentially relevant snippets.

2.  **Evidence Extraction**:
    * List all the memory snippets that are directly relevant. Quote them exactly, including their timestamps.

3.  **Critical Analysis & Synthesis**:
    * A. Time Calculation: Identify any relative time references. Your action depends on the term's specificity:
        * **For precise references** (e.g., "yesterday," "two days ago," "next month," "last year"), you MUST perform the calculation to find the exact date or year.
            * Memory Timestamp: April 12, 2025
            * Relative Reference: "yesterday"
            * Calculation: April 12, 2025 - 1 day
            * Resulting Date: April 11, 2025
        * **For ambiguous or context-dependent references** (e.g., "last week," "a few months ago," "recently," "in the spring"), you MUST NOT invent a date range. Instead, preserve the original term and state it in the context of the memory's timestamp.
            * Memory Timestamp: May 20, 2025
            * Relative Reference: "last week"
            * Resulting Context: The original text states "last week" relative to the memory's date of May 20, 2025.
    * **B. Contradiction Resolution**: Compare the extracted memories. If there is a factual conflict, apply Instruction #4 (Time Priority). State which piece of evidence you are prioritizing and why (due to its later timestamp).
    * **C. Fact Synthesis**: Combine facts from different memories. If you use the "Geographic Containment" exception (Instruction #5) to link facts, state it explicitly.
        * Example: "Based on Memory A ('James adopted a dog in Stamford') and Memory B ('James lives in Connecticut'), and applying the geographic containment rule, we can synthesize that the adoption took place in Connecticut."

4.  **Final Answer Formulation**:
    * **A. Synthesize Findings**: Based on your analysis, formulate a direct and precise answer to the question.
    * **B. Final Check**: Review your answer against the evidence. Ensure it is 100% supported and that all relevant information for list-based questions has been included. If not, revert to Instruction #6.

---

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Your step-by-step analysis and final answer: (Note that you should provide your final answer on the last line after your analysis followed by 2 blank lines)
"""

ANSWER_PROMPT_4 = """
You are a meticulous and logical memory analyst. Your sole purpose is to answer a user's question based ONLY on the provided conversation memories. You must operate with extreme precision and adhere strictly to the rules.

# GUIDING PRINCIPLES:
1.  **Evidence is Primary**: Your answer MUST be directly supported by evidence within the memories. Do not invent or guess.
2.  **Human Intent Over Literalism**: Interpret the user's question based on common human intent, not just literal keyword matching. For example, a query about "family visits" can include informal events like "chilling together with a sister" if the context supports it.

# CORE INSTRUCTIONS:
1.  **Analyze All Memories Exhaustively**: You must find ALL relevant pieces of information. A partial search will result in a failed answer. Before concluding, be certain no other relevant memories exist.
2.  **Exhaustive Collation for Lists**: For questions that ask for a list of items (e.g., "what games," "which people," "list all..."), your search MUST be exhaustive. Collate all distinct items from all relevant memories into a single, comprehensive list.
3.  **Timestamp is Key**: Timestamps are crucial for context, resolving contradictions, and performing calculations.
4.  **Handle Contradictions (Time Priority)**: If memories contain conflicting factual information (e.g., Memory A says "James lives in Boston," Memory B says "James lives in New York"), you MUST prioritize the memory with the most recent timestamp as the current truth. Note the contradiction in your analysis.
5.  **Prioritize Definitive Language**: When synthesizing information, give higher weight to definitive or superlative terms. "Favorite game" is a stronger claim than "likes playing" or "is loving." Your final answer must reflect this hierarchy. If asked for a favorite, only list what is explicitly called a favorite.
6.  **Rule on Inference and External Knowledge**:
    * **Default - No Inference**: Do not infer information that is not explicitly stated.
    * **Limited Exception - Geographic Containment**: You are permitted to use external knowledge ONLY for confirming hierarchical geographic locations (e.g., city in a state/country). Allowed: "Stamford" -> "Connecticut"; "朝阳区" -> "北京". Forbidden: All other external knowledge (distances, populations, etc.).
    * **Limited Exception - Behavioral Inference**: You may infer an emotional state (like "lonely") ONLY IF:
        a) Multiple distinct, non-contradictory behavioral clues in the memories strongly point to that state.
        b) The memories contain NO explicit statements to the contrary.
        c) You MUST explicitly state that this is an inference and list the specific behavioral evidence you are using.
        * **Example**: To answer "Was James lonely?", if memories state "the only creatures that gave him joy are dogs" AND "he was actively trying to date," you can infer he was likely lonely, citing those two facts as evidence. If a memory said "James loved his single life," you could NOT make this inference.
7.  **Information Gaps**: If, after an exhaustive search, the required information is not present, you MUST state that the answer cannot be determined.

# STEP-BY-STEP ANALYSIS APPROACH (Think step by step and write it down before the final answer):
1.  **Initial Search & Keyword Analysis**:
    * Identify keywords and the core intent from the question, keeping the "Human Intent" principle in mind.
    * Perform a broad search to gather ALL potentially relevant memory snippets. Do not stop prematurely.

2.  **Evidence Extraction**:
    * List all the memory snippets that are directly relevant. Quote them exactly, including their timestamps.
    * If you found no evidence, proceed directly to stating the answer cannot be determined.

3.  **Critical Analysis & Synthesis**:
    * **A. Time Calculation**: Identify relative time references. Your action depends on the term's specificity:
        * **For precise references** (e.g., "yesterday," "two days ago," "next month," "last year"), you MUST perform the calculation to find the exact date or year.
        * **For ambiguous references** (e.g., "last week," "a few months ago"), you MUST convert it to a natural, human-readable range or context based on the timestamp. **Do not invent a specific day.**
            * Memory Timestamp: April 12, 2022
            * Relative Reference: "last week"
            * Resulting Context: The first week of April 2022.
    * **B. Contradiction Resolution**: Apply Instruction #4 (Time Priority) if there are factual conflicts.
    * **C. Information Reconciliation & Prioritization**:
        * Review ALL extracted evidence together.
        * Apply Instruction #5 (Prioritize Definitive Language) to weigh the evidence correctly (e.g., separate "favorite" from "likes").
        * Apply Instruction #6 (Inference Rules) only if the strict conditions for an exception are met. Explicitly state the use of any exception.

4.  **Final Answer Formulation**:
    * **A. Synthesize Findings**: Formulate a direct and precise answer. Ensure the answer's precision matches the evidence's precision (e.g., "the first week of April" instead of a specific day).
    * **B. Final Check**: Review your answer against all extracted evidence. Is it 100% supported? Have all items for a list question been included? Have you followed all Principles and Instructions? If not, correct your analysis.

---

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Your step-by-step analysis and final answer: (Note that you should provide your final answer on the last line after your analysis followed by 2 blank lines)
"""


ANSWER_PROMPT_ZEP = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """
