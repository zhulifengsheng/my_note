# RAG

1. 检索  
    - 索引优化：
        - 固定长度切分
        - 根据语义进行切分
        - 使用默认分隔符（如。）进行切分
        - late chunking：之前是先chunk再embedding，语义向量局部在chunk中；而late chunking是先embedding再chunk，这样embedding的信息会更加充分(JINA)
        - 利用LLM将分块和完整文档一起传给LLM，LLM给chunk显示添加上下文信息(Anthropic)
    - 查询优化：
        - HyDE：第一步让LLM针对query生成多个假设文档。第二步对生成的多个假设文档进行向量编码之后，用它们的平均向量去检索最相似的文档。在计算平均向量时，也可以考虑将原始query的向量作为其输入向量的一部分。
        - BM25：BM25的核心思想是基于词频(TF)和逆文档频率(IDF)来，同时还引入了文档的长度信息来计算文档D和查询Q之间的相关性。
            - **词频**：这是查询中的词 q_i在文档 D 中出现的频率。
            - **逆文档频率**：一个词在很多文档中出现，其IDF值就会低，反之则高。这意味着罕见的词通常有更高的IDF值，从而在相关性评分中拥有更大的权重。
    - Embedding：
        - M3E：
        - BGE：
    - Reranker（精筛）：
        - 对多路召回的数据（粗选），进行归一化重排序。效果比Embedding的相似度强

2. 生成
    - query改写

# GRAPHRAG

