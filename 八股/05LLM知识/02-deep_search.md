# deep search
- [X] storm paper
- [ ] jina
- [ ] openai
- [ ] grok
- [ ] tencent
## 前身
### STORM
**research工作流**
pre-writing stage（搜索query相关的信息生成大纲） + writing stage（xxx）

**pre-writing stage**
1. 发现当前query（即topic）的多个视角【提升研究、搜索的全面性】
2. 模拟不同视角的wiki作者向专家（掌握可信源）提出问题 + 回答的对话【收集问答形式的信息】
3. 用收集的信息 + 草稿大纲 生成 新大纲  

评价方法：利用 NER + embedding相似度评价模型生成的大纲 和 真实大纲的差异
![大纲生成](storm.jpg)

**writing stage**
1. 每个章节单独生成正文
2. 利用**pre-writing stage**阶段收集的references，完成写作

**具体的流程**
1. topic -> related topic
2. related topic -> 相关wiki页面的目录
3. 相关wiki页面的目录 -> 多个视角（包括topic本身的基础视角）
4. 模拟生成wiki作者以某个视角和专家（可以搜索大量wiki页面）进行对话 -> （q1, a1, q2, a2, ..., qm, am） 【q由wiki作者生成】 
    a. query分解【由专家进行分解】  
    b. 专家利用URL进行可信源的判断，获取可信内容进行回答
    c. 可信源被添加到references（搜索到的网页正文）中，用于正文写作
4. 草稿大纲 + 对话 -> 优化大纲
5. 用embedding检索和各级title相关的文档（文档来自references）
6. 并行生成各章节的正文
7. 合并所有章节去除各章节中重复的部分


## 百家争鸣
### jina
123345567
789
> 123

### openai

### grok

## 业务实现
### 工作流

### 评估

### 训练

### 遇到的问题

### 探索

