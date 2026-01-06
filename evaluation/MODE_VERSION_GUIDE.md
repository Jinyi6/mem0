# 模式版本号查看指南

本文档说明如何查看 `fact_extraction_mode`、`memory_decision_mode`、`search_mode`、`answer_mode` 的具体版本号。

## 📍 查看位置

### 1. 配置文件位置

**主要配置文件**：`config/mem0_qwen.json`

在 `exp_params` 部分可以看到当前使用的版本号：

```json
{
  "exp_params": {
    "fact_extraction_mode": "5",
    "memory_decision_mode": "5",
    "search_mode": "6",
    "answer_mode": "5"
  }
}
```

### 2. 代码中的版本映射

#### 2.1 `fact_extraction_mode` - 事实提取模式

**文件位置**：`src/memzero_wo_client/add.py` (第 180-219 行)

**支持的版本号**：
- `"0"` → `FACT_RETRIEVAL_PROMPT`
- `"1"` → `FACT_RETRIEVAL_PROMPT_1`
- `"2"` → `FACT_RETRIEVAL_PROMPT_2`
- `"3"` → `FACT_RETRIEVAL_PROMPT_3`
- `"3c"` → `FACT_RETRIEVAL_PROMPT_3c`
- `"5"` → `FACT_RETRIEVAL_PROMPT_5`
- `"10"` → `FACT_RETRIEVAL_PROMPT_10`
- `"12"` → `FACT_RETRIEVAL_PROMPT_12`
- `"13"` → `FACT_RETRIEVAL_PROMPT_13`
- `"14"` → `FACT_RETRIEVAL_PROMPT_14`
- `"14.6"` → `FACT_RETRIEVAL_PROMPT_14_6`
- `"14.7"` → `FACT_RETRIEVAL_PROMPT_14_7`

**Prompt 定义位置**：`mem0/configs/prompts.py`

---

#### 2.2 `memory_decision_mode` - 记忆决策模式

**文件位置**：`src/memzero_wo_client/add.py` (第 221-265 行)

**支持的版本号**：
- `"0"` → `DEFAULT_UPDATE_MEMORY_PROMPT`
- `"0c"` → `UPDATE_MEMORY_PROMPT_0c`
- `"1"` → `UPDATE_MEMORY_PROMPT_1`
- `"2"` → `UPDATE_MEMORY_PROMPT_2`
- `"2agg"` → `UPDATE_MEMORY_PROMPT_2_agg`
- `"2con"` → `UPDATE_MEMORY_PROMPT_2_con`
- `"5"` → `UPDATE_MEMORY_PROMPT_5`
- `"10"` → `UPDATE_MEMORY_PROMPT_10`
- `"11"` → `UPDATE_MEMORY_PROMPT_11`
- `"12"` → `UPDATE_MEMORY_PROMPT_12`
- `"13"` → `UPDATE_MEMORY_PROMPT_13`
- `"14"` → `UPDATE_MEMORY_PROMPT_14`
- `"14.5"` → `UPDATE_MEMORY_PROMPT_14_5`
- `"14.6"` → `UPDATE_MEMORY_PROMPT_14_6`
- `"14.7"` → `UPDATE_MEMORY_PROMPT_14_7`

**Prompt 定义位置**：`mem0/configs/prompts.py`

---

#### 2.3 `search_mode` - 搜索模式

**文件位置**：`src/memzero_wo_client/search_async.py` (第 2199 行开始的 `Search()` 方法)

**支持的版本号**：
- `"3"` → 问题分解 + 多查询搜索 + Reranking (第 2217 行)
- `"5"` → PRF扩展 + Reranking + MMR多样化 (第 2349 行)
- `"6"` → (第 2798 行)
- `"10"` → (第 2833 行)
- `"14.5"` → (第 3076 行)
- `"14.6"` → (第 3080 行)
- `"14.7"` → (第 3084 行)
- `"14.8"` → (第 3088 行)
- `"14.9"` → PRF扩展 + Reranking + MMR多样化 (增强版) (第 2349 行)
- `"14.10"` → (第 2795 行)

**注意**：搜索模式的具体实现逻辑在 `Search()` 方法中，每个版本对应不同的搜索策略。

---

#### 2.4 `answer_mode` - 回答模式

**文件位置**：`src/memzero_wo_client/search_async.py` (第 373-428 行)

**支持的版本号**：
- `"0"` → `ANSWER_PROMPT` (默认)
- `"1"` → `ANSWER_PROMPT_1`
- `"2"` → `ANSWER_PROMPT_2`
- `"3"` → `ANSWER_PROMPT_3`
- `"4"` → `ANSWER_PROMPT_4`
- `"5"` → `ANSWER_PROMPT_5`
- `"6"` → `ANSWER_PROMPT_6`
- `"7"` → `ANSWER_PROMPT_7`
- `"10"` → `ANSWER_PROMPT_10`
- `"12"` → `ANSWER_PROMPT_12`
- `"13"` → `ANSWER_PROMPT_13`
- `"14"` → `ANSWER_PROMPT_14`
- `"14.6"` → `ANSWER_PROMPT_14_6`
- `"14.7"` → `ANSWER_PROMPT_14_7`
- `"14.8"` → `ANSWER_PROMPT_14_8`
- `"14.9"` → `ANSWER_PROMPT_14_9`

**Prompt 定义位置**：`src/memzero_wo_client/prompts.py`

---

## 🔍 如何查看当前使用的版本

### 方法1：查看配置文件

```bash
cat config/mem0_qwen.json | grep -A 10 "exp_params"
```

### 方法2：查看实验结果文件名

实验结果文件名包含版本号信息：
```
evaluation_metrics_{timestamp}_{fact_extraction_mode}_{memory_decision_mode}_{search_mode}_{answer_mode}.json
```

例如：
```
evaluation_metrics_20251230_132739_5_5_6_5.json
```
表示：
- `fact_extraction_mode`: 5
- `memory_decision_mode`: 5
- `search_mode`: 6
- `answer_mode`: 5

### 方法3：查看代码中的映射

直接查看以下文件中的 if-elif 语句：
- `src/memzero_wo_client/add.py` (第 180-265 行) - fact_extraction_mode 和 memory_decision_mode
- `src/memzero_wo_client/search_async.py` (第 373-428 行) - answer_mode
- `src/memzero_wo_client/search_async.py` (第 2199 行开始) - search_mode

---

## 📝 修改版本号

### 修改配置文件

编辑 `config/mem0_qwen.json`：

```json
{
  "exp_params": {
    "fact_extraction_mode": "5",      // 修改这里
    "memory_decision_mode": "5",      // 修改这里
    "search_mode": "6",               // 修改这里
    "answer_mode": "5"                // 修改这里
  }
}
```

### 添加新版本

1. **添加新的 prompt 定义**：
   - fact_extraction_mode → `mem0/configs/prompts.py`
   - memory_decision_mode → `mem0/configs/prompts.py`
   - answer_mode → `src/memzero_wo_client/prompts.py`

2. **在代码中添加映射**：
   - `add.py` 中添加 fact_extraction_mode 和 memory_decision_mode 的映射
   - `search_async.py` 中添加 answer_mode 的映射
   - `search_async.py` 的 `Search()` 方法中添加 search_mode 的实现

---

## 📚 相关文件列表

- **配置文件**：`config/mem0_qwen.json`
- **事实提取和记忆决策**：`src/memzero_wo_client/add.py`
- **搜索和回答**：`src/memzero_wo_client/search_async.py`
- **Prompt 定义**：
  - `mem0/configs/prompts.py` (fact_extraction_mode, memory_decision_mode)
  - `src/memzero_wo_client/prompts.py` (answer_mode)

