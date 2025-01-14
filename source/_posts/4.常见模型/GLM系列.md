## chatglm是属于哪个类型
ChatGLM属于Prefix Decoder类型。LLM中的三大主流框架：Causal Decoder、Prefix Decoder及Encoder-Decoder。
### Causal Decoder
1. 结构特点
Causal Decoder，又称因果语言模型，其典型代表为GPT系列模型。该框架采用从左到右的单向注意力机制，确保每个输入token只能注意到过去的token和它本身。这种自回归（Auto Regressive）的方式使得模型能够根据历史信息逐步生成文本。

2. 优点
- 训练效率高：Causal Decoder在所有token上计算损失，充分利用了训练数据，提高了训练效率。
- Zero-shot能力强：由于遵循严格的单向注意力规则，模型在零样本学习（Zero-shot Learning）任务中表现出色。
- 涌现能力：随着模型规模的增大，Causal Decoder能够展现出一些令人惊讶的涌现能力（Emergent Abilities），如创作小说、编写代码等。
  
3. 适用场景
Causal Decoder适用于文本生成任务，如对话生成、文本续写、文章创作等。

### Prefix Decoder
1. 结构特点
Prefix Decoder，即前缀语言模型，其结构介于Causal Decoder和Encoder-Decoder之间。该框架在输入部分采用双向注意力，允许前缀序列中的任意两个token相互可见；而在输出部分则采用单向注意力，类似于Causal Decoder。代表模型有ChatGLM、U-PaLM等。

2. 优点
输入理解充分：由于输入部分采用双向注意力，Prefix Decoder对问题的编码理解更为充分。
输出控制灵活：输出部分的单向注意力机制使得模型在生成文本时能够遵循一定的逻辑顺序。
3. 缺点
训练效率低：相比于Causal Decoder，Prefix Decoder在训练时只会在输出上计算损失，导致训练效率较低。
4. 适用场景
Prefix Decoder适用于需要同时考虑输入理解和输出控制的场景，如问答系统、文本摘要等。

### Encoder-Decoder
1. 结构特点
Encoder-Decoder是Transformer模型最初提出时采用的架构，由独立的Encoder和Decoder两部分组成。Encoder将输入序列处理为一种中间表示，而Decoder则基于该中间表示自回归地生成目标序列。代表模型有T5、Flan-T5等。

2. 优点
输入理解深入：Encoder部分采用双向注意力，对输入序列的编码理解非常深入。
输出生成灵活：Decoder部分基于Encoder的中间表示生成目标序列，输出生成过程灵活多样。
3. 缺点
长文本生成效果差：在长文本生成任务上，Encoder-Decoder架构的效果往往不如Causal Decoder和Prefix Decoder。
训练效率低：由于模型结构相对复杂，训练效率也相对较低。
4. 适用场景
Encoder-Decoder适用于需要深入理解输入并生成复杂输出的场景，如机器翻译、文本摘要等。