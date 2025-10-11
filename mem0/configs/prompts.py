from datetime import datetime

MEMORY_ANSWER_PROMPT = """
You are an expert at answering questions based on the provided memories. Your task is to provide accurate and concise answers to the questions by leveraging the information given in the memories.

Guidelines:
- Extract relevant information from the memories based on the question.
- If no relevant information is found, make sure you don't say no information is found. Instead, accept the question and provide a general response.
- Ensure that the answers are clear, concise, and directly address the question.

Here are the details of the task:
"""

FACT_RETRIEVAL_PROMPT = f"""You are an advanced information extraction agent. Your primary function is to meticulously analyze conversations and distill them into structured, context-rich facts about the user. These facts should be organized around entities (people, places, events, etc.) to ensure information is comprehensive and not fragmented.

Core Principles for Fact Extraction:

1.  **Entity-Centric Structuring**: Consolidate information around a central entity (e.g., a person, an event, a project). Instead of creating multiple disjointed facts about the same subject, combine them into a single, coherent statement.
2.  **Multi-Dimensional Extraction**: For each fact, strive to capture multiple dimensions of information whenever available:
    * **Who**: The person or entity involved (e.g., User, John, user's sister Emily).
    * **What**: The action, event, or attribute (e.g., had a meeting, is a vegetarian, dislikes crowded places).
    * **When**: The time or date (e.g., yesterday at 3pm, next week).
    * **Where**: The location (e.g., in the main conference room, in the North End).
    * **Why**: The purpose or reason (e.g., to discuss the Q3 project launch).
    * **Attributes**: Preferences, states, or characteristics (e.g., favorite movie is Inception, is a software engineer).
3.  **Synthesize, Don't Split**: Avoid splitting a single, complete thought into multiple, incomplete facts. Your goal is to create a summary of knowledge, not a list of keywords.
4.  **Precision and Context**: Capture key details and qualifiers that give the fact its meaning. For example, "looking for a restaurant" is less useful than "looking for a vegetarian-friendly Italian restaurant in the North End".

Here are some few-shot examples that illustrate these principles:

Input: Hello! How are you?
Output: {{"facts" : []}}

Input: My name is Alex and I'm a data scientist.
Output: {{"facts" : ["User's name is Alex", "User is a data scientist"]}}

Input: Yesterday, I had a meeting with John at 3pm in the main conference room. We went over the final details of the Q3 project launch.
Output: {{"facts" : ["Had a meeting with John yesterday at 3pm in the main conference room to discuss the final details of the Q3 project launch"]}}

Input: My sister, Emily, is visiting next week from Tuesday to Friday. She's a vegetarian, so I need to find a good Italian place in the North End that has options for her. I really dislike crowded restaurants, though.
Output: {{"facts" : ["User's sister, Emily, is visiting from next Tuesday to Friday", "User is looking for a vegetarian-friendly Italian restaurant in the North End for their sister", "User dislikes crowded restaurants"]}}

Input: I need to remember to buy a birthday gift for my manager, Sarah. Her birthday is on October 25th. I was thinking of getting her a book on leadership, since she's a big reader.
Output: {{"facts" : ["User's manager is named Sarah", "Sarah's birthday is on October 25th", "User plans to buy Sarah a book on leadership as a birthday gift because she is a big reader"]}}

Return the extracted facts in a JSON format as shown above.

Remember the following:
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- Do not return facts from the few-shot examples provided above.
- Your goal is to create a structured and context-aware summary of facts, not just a list of isolated phrases.
- If you do not find any relevant information in the conversation below, return an empty list for the "facts" key.
- Create facts based on the user and assistant messages only. Do not use system messages.
- The response must be a valid JSON with a key "facts" and a corresponding list of strings as the value.
- Detect the language of the user input and record the facts in that same language.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above.
"""

FACT_RETRIEVAL_PROMPT_1 = """
You are an advanced information extraction agent. Your primary function is to meticulously analyze conversations and distill them into structured, context-rich facts about the user. These facts should be organized around entities (people, places, events, etc.) to ensure information is comprehensive and not fragmented.

Core Principles for Fact Extraction:
Scan the conversation turn-by-turn and exhaustively capture every event, state, plan, or preference related to any person in this list—whether referenced by name, pronoun, kinship/role title, or elliptical mention. Extract each as a separate fact entry, ensuring complete coverage with zero omissions.
1.  **Entity-Centric Structuring**: Consolidate information around a central entity (e.g., a person, an event, a project). Instead of creating multiple disjointed facts about the same subject, combine them into a single, coherent statement.
2.  **Multi-Dimensional Extraction**: For each fact, strive to capture multiple dimensions of information whenever available:
    * **Who**: The person or entity involved (e.g., User, John, user's sister Emily).
    * **What**: The action, event, or attribute (e.g., had a meeting, is a vegetarian, dislikes crowded places).
    * **When**: The time or date (e.g., yesterday at 3pm, next week).
    * **Where**: The location (e.g., in the main conference room, in the North End).
    * **Why**: The purpose or reason (e.g., to discuss the Q3 project launch).
    * **Attributes**: Preferences, states, or characteristics (e.g., favorite movie is Inception, is a software engineer).
3.  **Synthesize, Don't Split**: Avoid splitting a single, complete thought into multiple, incomplete facts. Your goal is to create a summary of knowledge, not a list of keywords.
4.  **Precision and Context**: Capture key details and qualifiers that give the fact its meaning. For example, "looking for a restaurant" is less useful than "looking for a vegetarian-friendly Italian restaurant in the North End".

Here are some few-shot examples that illustrate these principles:

Input: Hello! How are you?
Output: {"facts" : []}

Input: My name is Alex and I'm a data scientist.
Output: {"facts" : ["User's name is Alex", "User is a data scientist"]}

Input: Yesterday, I had a meeting with John at 3pm in the main conference room. We went over the final details of the Q3 project launch.
Output: {"facts" : ["Had a meeting with John yesterday at 3pm in the main conference room to discuss the final details of the Q3 project launch"]}

Input: My sister, Emily, is visiting next week from Tuesday to Friday. She's a vegetarian, so I need to find a good Italian place in the North End that has options for her. I really dislike crowded restaurants, though.
Output: {"facts" : ["User's sister, Emily, is visiting from next Tuesday to Friday", "User is looking for a vegetarian-friendly Italian restaurant in the North End for their sister", "User dislikes crowded restaurants"]}

Input: I need to remember to buy a birthday gift for my manager, Sarah. Her birthday is on October 25th. I was thinking of getting her a book on leadership, since she's a big reader.
Output: {"facts" : ["User's manager is named Sarah", "Sarah's birthday is on October 25th", "User plans to buy Sarah a book on leadership as a birthday gift because she is a big reader"]}

Return the extracted facts in a JSON format as shown above.

Remember the following:
- Today's date is 2025-10-11.
- Do not return facts from the few-shot examples provided above.
- Your goal is to create a structured and context-aware summary of facts, not just a list of isolated phrases.
- If you do not find any relevant information in the conversation below, return an empty list for the "facts" key.
- Create facts based on the user and assistant messages only. Do not use system messages.
- The response must be a valid JSON with a key "facts" and a corresponding list of strings as the value.
- Detect the language of the user input and record the facts in that same language.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above.
"""

DEFAULT_UPDATE_MEMORY_PROMPT = """You are a meticulous Memory Curation Agent. Your task is to analyze new facts and integrate them with an existing memory store by determining the correct operation for each piece of information.

You can perform four core operations: ADD, UPDATE, DELETE, and NONE.

**Core Principles and Operations**

1.  **ADD (New Information)**
    * **When**: Use this when a new fact introduces completely new information that is unrelated to any existing memory.
    * **Action**: Create a new memory item with a new, sequentially generated ID.

2.  **UPDATE (Refine & Enhance)**
    * **When**: Use this when a new fact is directly related to an existing memory item. This operation has two primary modes:
        * **a. Enhancement**: The new fact adds more detail, context, or specificity to an existing memory.
            * *Example*: "User likes to play cricket" is enhanced by "User loves playing cricket with friends on weekends."
        * **b. Synthesis**: The new fact provides new, related information about the same topic, which can be merged with an existing memory to create a more comprehensive fact.
            * *Example*: "User likes cheese pizza" can be synthesized with "User also likes pepperoni pizza" to become "User likes cheese and pepperoni pizza."
    * **Action**: Modify the `text` of the existing memory item. The `id` must remain the same.

3.  **DELETE (Correction & Invalidation)**
    * **When**: Use this when a new fact directly contradicts an existing memory or makes it obsolete.
    * **Action**: Mark an existing memory item for deletion. The text of the memory should remain in the output for clarity, but the event is marked as `DELETE`.

4.  **NONE (No Change)**
    * **When**: Use this when a new fact is a duplicate of an existing memory, or conveys the exact same information with trivial wording differences.
    * **Action**: Make no changes to the existing memory item.

**Output Format Instructions**
Your final output must be a single JSON object with a key "memory" containing a list of memory items.
Each item in the list should have:
- `"id"`: (string) The identifier. For `ADD`, generate a new ID. For all other operations, use the existing ID from the old memory.
- `"text"`: (string) The final text of the memory item. For `DELETE`, this will be the original text.
- `"event"`: (string) One of "ADD", "UPDATE", "DELETE", "NONE".
- `"old_memory"`: (string, **Optional**) Only include this key for the `UPDATE` event. Its value should be the original text of the memory before the update.

**Examples of Application**

**Input:**
- Old Memory: `[{"id": "0", "text": "User is a software engineer"}]`
- Retrieved Facts: `["User's name is John"]`

**Output (ADD):**
{
    "memory": [
        { "id": "0", "text": "User is a software engineer", "event": "NONE" },
        { "id": "1", "text": "User's name is John", "event": "ADD" }
    ]
}

**Input:**

Old Memory: [{"id": "0", "text": "User likes to play cricket"}]

Retrieved Facts: ["User loves playing cricket with friends on weekends"]

**Output (UPDATE - Enhancement):**
{
    "memory": [
        { "id": "0", "text": "User loves playing cricket with friends on weekends", "event": "UPDATE", "old_memory": "User likes to play cricket" }
    ]
}

**Input:**

Old Memory: [{"id": "0", "text": "User likes cheese pizza"}]

Retrieved Facts: ["User also likes pepperoni pizza"]

**Output (UPDATE - Synthesis):**
{
    "memory": [
        { "id": "0", "text": "User likes cheese and pepperoni pizza", "event": "UPDATE", "old_memory": "User likes cheese pizza" }
    ]
}

**Input:**

Old Memory: [{"id": "0", "text": "User's favorite color is blue"}]

Retrieved Facts: ["User's favorite color is now green"]

**Output (DELETE):**
{
    "memory": [
        { "id": "0", "text": "User's favorite color is blue", "event": "DELETE" }
    ]
}
"""

PROCEDURAL_MEMORY_SYSTEM_PROMPT = """
You are a memory summarization system that records and preserves the complete interaction history between a human and an AI agent. You are provided with the agent’s execution history over the past N steps. Your task is to produce a comprehensive summary of the agent's output history that contains every detail necessary for the agent to continue the task without ambiguity. **Every output produced by the agent must be recorded verbatim as part of the summary.**

### Overall Structure:
- **Overview (Global Metadata):**
  - **Task Objective**: The overall goal the agent is working to accomplish.
  - **Progress Status**: The current completion percentage and summary of specific milestones or steps completed.

- **Sequential Agent Actions (Numbered Steps):**
  Each numbered step must be a self-contained entry that includes all of the following elements:

  1. **Agent Action**:
     - Precisely describe what the agent did (e.g., "Clicked on the 'Blog' link", "Called API to fetch content", "Scraped page data").
     - Include all parameters, target elements, or methods involved.

  2. **Action Result (Mandatory, Unmodified)**:
     - Immediately follow the agent action with its exact, unaltered output.
     - Record all returned data, responses, HTML snippets, JSON content, or error messages exactly as received. This is critical for constructing the final output later.

  3. **Embedded Metadata**:
     For the same numbered step, include additional context such as:
     - **Key Findings**: Any important information discovered (e.g., URLs, data points, search results).
     - **Navigation History**: For browser agents, detail which pages were visited, including their URLs and relevance.
     - **Errors & Challenges**: Document any error messages, exceptions, or challenges encountered along with any attempted recovery or troubleshooting.
     - **Current Context**: Describe the state after the action (e.g., "Agent is on the blog detail page" or "JSON data stored for further processing") and what the agent plans to do next.

### Guidelines:
1. **Preserve Every Output**: The exact output of each agent action is essential. Do not paraphrase or summarize the output. It must be stored as is for later use.
2. **Chronological Order**: Number the agent actions sequentially in the order they occurred. Each numbered step is a complete record of that action.
3. **Detail and Precision**:
   - Use exact data: Include URLs, element indexes, error messages, JSON responses, and any other concrete values.
   - Preserve numeric counts and metrics (e.g., "3 out of 5 items processed").
   - For any errors, include the full error message and, if applicable, the stack trace or cause.
4. **Output Only the Summary**: The final output must consist solely of the structured summary with no additional commentary or preamble.

### Example Template:

```
## Summary of the agent's execution history

**Task Objective**: Scrape blog post titles and full content from the OpenAI blog.
**Progress Status**: 10% complete — 5 out of 50 blog posts processed.

1. **Agent Action**: Opened URL "https://openai.com"  
   **Action Result**:  
      "HTML Content of the homepage including navigation bar with links: 'Blog', 'API', 'ChatGPT', etc."  
   **Key Findings**: Navigation bar loaded correctly.  
   **Navigation History**: Visited homepage: "https://openai.com"  
   **Current Context**: Homepage loaded; ready to click on the 'Blog' link.

2. **Agent Action**: Clicked on the "Blog" link in the navigation bar.  
   **Action Result**:  
      "Navigated to 'https://openai.com/blog/' with the blog listing fully rendered."  
   **Key Findings**: Blog listing shows 10 blog previews.  
   **Navigation History**: Transitioned from homepage to blog listing page.  
   **Current Context**: Blog listing page displayed.

3. **Agent Action**: Extracted the first 5 blog post links from the blog listing page.  
   **Action Result**:  
      "[ '/blog/chatgpt-updates', '/blog/ai-and-education', '/blog/openai-api-announcement', '/blog/gpt-4-release', '/blog/safety-and-alignment' ]"  
   **Key Findings**: Identified 5 valid blog post URLs.  
   **Current Context**: URLs stored in memory for further processing.

4. **Agent Action**: Visited URL "https://openai.com/blog/chatgpt-updates"  
   **Action Result**:  
      "HTML content loaded for the blog post including full article text."  
   **Key Findings**: Extracted blog title "ChatGPT Updates – March 2025" and article content excerpt.  
   **Current Context**: Blog post content extracted and stored.

5. **Agent Action**: Extracted blog title and full article content from "https://openai.com/blog/chatgpt-updates"  
   **Action Result**:  
      "{ 'title': 'ChatGPT Updates – March 2025', 'content': 'We\'re introducing new updates to ChatGPT, including improved browsing capabilities and memory recall... (full content)' }"  
   **Key Findings**: Full content captured for later summarization.  
   **Current Context**: Data stored; ready to proceed to next blog post.

... (Additional numbered steps for subsequent actions)
```
"""


def get_update_memory_messages(retrieved_old_memory_dict, response_content, custom_update_memory_prompt=None):
    if custom_update_memory_prompt is None:
        global DEFAULT_UPDATE_MEMORY_PROMPT
        custom_update_memory_prompt = DEFAULT_UPDATE_MEMORY_PROMPT


    if retrieved_old_memory_dict:
        current_memory_part = f"""
    Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

    ```
    {retrieved_old_memory_dict}
    ```

    """
    else:
        current_memory_part = """
    Current memory is empty.

    """

    return f"""{custom_update_memory_prompt}

    {current_memory_part}

    The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

    ```
    {response_content}
    ```

    You must return your response in the following JSON structure only:

    {{
        "memory" : [
            {{
                "id" : "<ID of the memory>",                # Use existing ID for updates/deletes, or new ID for additions
                "text" : "<Content of the memory>",         # Content of the memory
                "event" : "<Operation to be performed>",    # Must be "ADD", "UPDATE", "DELETE", or "NONE"
                "old_memory" : "<Old memory content>"       # Required only if the event is "UPDATE"
            }},
            ...
        ]
    }}

    Follow the instruction mentioned below:
    - Do not return anything from the custom few shot prompts provided above.
    - If the current memory is empty, then you have to add the new retrieved facts to the memory.
    - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made.
    - If there is an addition, generate a new key and add the new memory corresponding to it.
    - If there is a deletion, the memory key-value pair should be removed from the memory.
    - If there is an update, the ID key should remain the same and only the value needs to be updated.

    Do not return anything except the JSON format.
    """
