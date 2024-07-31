# 第八章：模型部署优化
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# 问题与解答
_问：剪枝如何提高模型效率？_

答：剪枝通过移除不太重要的神经元来减少模型的大小，这可以缩短推理时间并减少内存使用量，而不会显著影响模型的性能。

_问：什么是使用 GPTQ 进行训练后量化？_

答：使用 GPTQ（广义泊松训练量化）进行训练后量化涉及在训练后降低模型参数的精度，这可以减少模型大小并加快执行速度，而不会造成明显的精度损失。

_问：A/B 测试和影子部署在部署策略方面有何不同？_

答：A/B 测试涉及将一部分流量定向到新模型，以将其性能与现有模型进行比较。相比之下，影子部署是在不将真实用户流量定向到新模型的情况下，使其与现有模型并行运行，主要用于测试和评估目的。

_问：模型部署优化如何影响整体性能和可扩展性？_

答：模型部署中的优化（例如模型压缩、高效的硬件利用率和负载均衡）可以显著提高性能、降低成本并确保可扩展性以处理不同的负载。

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
