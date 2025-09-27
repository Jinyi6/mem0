# 测试locomo数据集：

第一步：add

```nohup python run_experiments.py --technique_type mem0 --method add --qdrant_path evaluation/qdrant_data/tmp/membench01  > logs/0927_membench_add.log 2>&1 &```

得到 memory

第二步：search

```nohup python run_experiments.py --technique_type mem0 --method search > logs/0926_locomo_search.log 2>&1 &```

得到结果，关于q & response的json

第三步：进行评估，计算指标

``` python evals.py --input_file results/{result}.json --output_file case_results.json```

统计平均得分

运行```evaluation/generate_scores.py```文件

# 关键文件
- 测试数据集：  ```evaluation/dataset```
- 实验数据存档  ```evaluation/results```    # 指的是已经确认跑完的数据
- memory存档    ```evaluation/exp_data/Memory```
- 实验数据目录  ```evaluation/exp_data```
- 向量化记忆    ```evaluation/qdrant_data```    # 运行```add```的时候记得修改目录

# 代码核心逻辑 - memory class

- **add() 方法 (当 infer=True 时):**

1. 第一次LLM调用 (事实提取):

接收一段原始对话，调用LLM。目的是从对话中提取出核心的事实或信息点 (new_retrieved_facts)，过滤掉口语化的、不重要的内容。

Prompt: 使用 get_fact_retrieval_messages 构建的prompt。

相似性搜索 (无LLM):对上一步提取出的每个“新事实”，去向量数据库中搜索已存在的、与之相似的“旧记忆”。

2. 第二次LLM调用 (记忆决策):

将找到的“旧记忆”和“新事实”一起发给LLM。让LLM扮演一个记忆管理员的角色，判断这些新事实应该如何处理。LLM会返回一个包含具体操作的JSON，比如：

- "event": "ADD": 这是一个全新的事实，需要新增。
- "event": "UPDATE": 新事实是对旧记忆的补充或修正，需要更新。
- "event": "DELETE": 新事实表明旧记忆已过时，需要删除。
- "event": "NONE": 无需操作。

Prompt: 使用 get_update_memory_messages 构建的prompt。

3. 执行操作:

代码解析LLM返回的JSON，并调用底层的 _create_memory, _update_memory, _delete_memory 方法来操作向量数据库。

- **search() 方法:**

这个过程相对简单，不涉及LLM。它接收文本查询，使用嵌入模型将其转换为向量，然后在向量数据库中执行相似度搜索。

- **记忆的增删改查 (CRUD) 逻辑**

1. 增 (Create)

触发: 在 add.py 中调用 memory.add()。

逻辑: 当LLM在“记忆决策”阶段判断一个新提取的事实是全新的信息时，会返回 ADD 指令。系统随后会为这条新记忆生成向量嵌入，并将其存入Qdrant数据库。

2. 删 (Delete)

触发:

主动删除: add.py 在处理每段对话前，会调用 memory.delete_all() 清空该用户的旧记忆。

智能删除: 在 memory.add() 过程中，如果LLM判断新信息使得旧记忆失效（例如，“会议改到下午4点了”这条信息使“会议在下午3点”的记忆失效），它会返回 DELETE 指令，系统则会从数据库中移除旧记忆。

3. 改 (Update)

触发: 在 memory.add() 过程中。

逻辑: 如果LLM判断新信息是对旧记忆的补充或更正（例如，从“我喜欢爬山”更新为“我喜欢爬山和游泳”），它会返回 UPDATE 指令。系统会用新的内容替换旧的记忆，并重新计算其向量嵌入。

4. 查 (Read/Search)

触发: 在 search.py 中调用 memory.search()。

逻辑: 将查询问题转换成向量，在Qdrant数据库中进行语义相似度搜索，返回最匹配的一组记忆。这些记忆随后被用作上下文，供另一个LLM调用来生成最终答案。


## LLM的使用

1. add.py 没有直接调用LLM。它通过调用 self.memory.add() 方法，将LLM的复杂交互委托给了 mem0 库（即 Memory.py）。LLM在这里的隐式作用是：接收原始对话文本，根据内部的prompt（比如脚本开头的 custom_instructions，虽然被注释了）智能提取和生成结构化的记忆点。

2. search.py 在 answer_question 方法中，通过 openai_client 明确地调用了LLM。这里的LLM扮演的是一个阅读理解和信息整合的角色。它的任务不是从自己的知识库里回答问题，而是根据提供给它的上下文（即检索出的记忆），来综合、推理并生成问题的答案。

