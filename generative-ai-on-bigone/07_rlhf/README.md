# 第七章：基于人类反馈的强化学习 (RLHF)
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# 问题与解答

_问：在强化学习的背景下，什么是人类对齐的概念？_

答：强化学习中的人类对齐是使模型的输出与人类价值观和偏好保持一致的过程，确保模型的输出有益、诚实且无害。

_问：强化学习如何促进生成式人工智能模型的微调？_

答：基于人类反馈的强化学习 (RLHF) 通过修改生成式模型的底层权重来对其进行微调，使其更好地与人类偏好保持一致，这些偏好通过奖励模型来表达。

_问：人在回路如何影响模型训练？_

答：人在回路通过提供人类反馈来影响模型训练，其中人类标注员对给定提示的各种补全进行排序，为每个提示创建多行训练数据。

_问：训练自定义奖励模型涉及哪些步骤？_

答：要训练自定义奖励模型，第一步是收集关于什么是“有益、诚实和无害”的人类反馈。这通常涉及使用 Amazon SageMaker Ground Truth 等托管服务来收集来自人类标注员的数据。

_问：Amazon SageMaker Ground Truth 在训练奖励模型中扮演什么角色？_

答：Amazon SageMaker Ground Truth 用于收集来自人类标注员的数据，允许他们对给定提示的补全进行排序，这对于训练奖励模型至关重要。

_问：Meta 的毒性检测器如何充当奖励模型？_

答：Meta 的毒性检测器基于 RoBERTa，通过预测给定文本输入在“非仇恨”或“仇恨”这两个类别上的概率分布来充当奖励模型。

_问：近端策略优化算法如何在 RLHF 中提供帮助？_

答：近端策略优化 (PPO) 算法通过根据奖励模型返回的奖励值更新生成式模型的权重，优化策略以生成与人类价值观一致的补全，从而在 RLHF 中提供帮助。

_问：如何在 RLHF 中减轻奖励黑客攻击？_

答：可以通过使用权重冻结的复制模型作为参考来减轻 RLHF 中的奖励黑客攻击。使用 KL 散度将 RLHF 微调模型的补全与该参考模型进行比较，以量化和控制差异。

_问：使用哪些方法来评估 RLHF 微调模型？_

答：RLHF 微调模型使用定性和定量评估技术进行评估，这涉及比较 RLHF 前后的提示补全。通常，毒性检测器用于使用来自 AllenAI 研究所的 RealToxicityPrompts 等毒性提示-补全数据集来评估模型。

# 章节
* [第一章](/01_intro) - 生成式人工智能用例、基础知识、项目生命周期
* [第二章](/02_prompt) - 提示工程和上下文学习
* [第三章](/03_foundation) - 大型语言基础模型
* [第四章](/04_optimize) - 量化和分布式计算
* [第五章](/05_finetune) - 微调和评估
* [第六章](/06_peft) - 参数高效微调 (PEFT)
* [第七章](/07_rlhf) - 使用带有 RLHF 的强化学习进行微调
* [第八章](/08_deploy) - 优化和部署生成式人工智能应用程序
* [第九章](/09_rag) - 检索增强生成 (RAG) 和代理
* [第十章](/10_multimodal) - 多模态基础模型
* [第十一章](/11_diffusers) - 使用 Stable Diffusion 进行受控生成和微调
* [第十二章](/12_bedrock) - 用于生成式人工智能的 Amazon Bedrock 托管服务

# 相关资源
* YouTube 频道：https://youtube.generativeaionaws.com
* 生成式人工智能 AWS Meetup（全球，虚拟）：https://meetup.generativeaionaws.com
* AWS 上的生成式人工智能 O'Reilly 图书：https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/
* AWS 上的数据科学 O'Reilly 图书：https://www.amazon.com/Data-Science-AWS-End-End/dp/1492079391/
