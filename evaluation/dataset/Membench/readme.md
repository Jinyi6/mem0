# 数据集

这些数据集主要沿着两个轴心进行分类：

交互场景 (Interaction Scenario):

参与式场景 (Participation Scenario, PS): 智能体以第一人称视角，主动与用户进行对话。数据形式是多轮对话。

观察式场景 (Observation Scenario, OS): 智能体以第三人称视角，被动地观察并记录用户输入的消息，不作回应。数据形式是消息列表。

记忆层次 (Memory Level):

事实性记忆 (Factual Memory, FM): 指对明确信息的底层记忆，例如姓名、日期、事件等具体事实。

反思性记忆 (Reflective Memory, RM): 指需要通过推理和归纳形成的高层记忆，例如从用户多次对不同菜肴的讨论中，总结出他“咸甜口味”的偏好。

将这两个维度组合起来，就得到了四种主要的数据集类型（如论文表2所示）：

PS-RM: 参与式场景 - 反思性记忆

PS-FM: 参与式场景 - 事实性记忆

OS-RM: 观察式场景 - 反思性记忆

OS-FM: 观察式场景 - 事实性记忆

FirstAgentHighLevel -> 参与式-反思性 (Participation-Reflective)

FirstAgentLowLevel -> 参与式-事实性 (Participation-Factual)

ThirdAgentHighLevel -> 观察式-反思性 (Observation-Reflective)

ThirdAgentLowLevel -> 观察式-事实性 (Observation-Factual)

---
"Single-hop": 单跳查询。回答问题只需要对话中的一条信息。

"Multi-hop": 多跳查询。需要结合对话中的多条分散信息才能回答。

"Comparative": 比较题。需要比较两个实体的某个共同属性（例如谁更年长）。

"Aggregative": 聚合/统计题。需要统计或汇总多个实体的信息（例如有多少人住在某个城市）。

"Knowledge_updating": 知识更新题。测试当新信息出现时，智能体是否能更新旧的、错误的记忆。

"Post_processing": 后处理题。需要对检索到的信息进行简单的推理或处理才能得出答案。

"Single-session-assistant": 单会话-助手记忆。测试对助手自己在单轮会话中说过的话的记忆。

"Multi-session-assistant": 多会话-助手记忆。测试对助手自己跨越多轮会话说过的话的记忆。

# 测试集
测试集。

子数据集1 (Sub-dataset 1, 普通规模): 从生成的数据中采样了总共880个问题（参与式场景120个RM + 360个FM，观察式场景60个RM + 280个FM）。

子数据集2 (Sub-dataset 2, 大规模, 约10万Token): 采样了总共219个问题（参与式场景30个RM + 90个FM，观察式场景15个RM + 84个FM）。


我的测试集

每个分类下的题目数量:
  - fh_Emotion_Emotion                     : 30 条
  - fh_Preference_book                     : 30 条
  - fh_Preference_food                     : 30 条
  - fh_Preference_movie                    : 30 条
  - fl_Aggregative_events                  : 23 条
  - fl_Aggregative_roles                   : 23 条
  - fl_Comparative_events                  : 23 条
  - fl_Comparative_roles                   : 23 条
  - fl_Knowledge_updating_events           : 23 条
  - fl_Knowledge_updating_roles            : 23 条
  - fl_Multi-hop_events                    : 23 条
  - fl_Multi-hop_roles                     : 23 条
  - fl_Multi-session-assistant_multi_agent : 23 条
  - fl_Post_processing_events              : 23 条
  - fl_Post_processing_roles               : 23 条
  - fl_Single-hop_events                   : 23 条
  - fl_Single-hop_roles                    : 23 条
  - fl_Single-session-assistant_book       : 23 条
  - fl_Single-session-assistant_food       : 23 条
  - fl_Single-session-assistant_movie      : 23 条
  - th_Emotion_Emotion                     : 15 条
  - th_Preference_book                     : 15 条
  - th_Preference_food                     : 15 条
  - th_Preference_movie                    : 15 条
  - tl_Aggregative_events                  : 12 条
  - tl_Aggregative_hybrid                  : 12 条
  - tl_Aggregative_roles                   : 12 条
  - tl_Comparative_events                  : 12 条
  - tl_Comparative_hybrid                  : 12 条
  - tl_Comparative_roles                   : 12 条
  - tl_Multi-hop_events                    : 12 条
  - tl_Multi-hop_hybrid                    : 12 条
  - tl_Multi-hop_items                     : 12 条
  - tl_Multi-hop_places                    : 12 条
  - tl_Multi-hop_roles                     : 12 条
  - tl_Single-hop_events                   : 12 条
  - tl_Single-hop_hybrid                   : 12 条
  - tl_Single-hop_items                    : 12 条
  - tl_Single-hop_places                   : 12 条
  - tl_Single-hop_roles                    : 12 条
  - tl_knowledge_updating_events           : 12 条
  - tl_knowledge_updating_roles            : 12 条
  - tl_post_processing_events              : 12 条
  - tl_post_processing_hybrid              : 12 条
  - tl_post_processing_items               : 12 条
  - tl_post_processing_places              : 12 条
  - tl_post_processing_roles               : 12 条
------------------------------------------------
  - 总计                                     : 824 条


每个分类下的题目数量:
  - Emotion     : 45 条
  - book        : 68 条
  - events      : 210 条
  - food        : 68 条
  - hybrid      : 60 条
  - items       : 36 条
  - movie       : 68 条
  - multi_agent : 23 条
  - places      : 36 条
  - roles       : 210 条
---------------------
  - 总计          : 824 条

比例基本匹配论文所报告的比例