# 第十一章：使用 Stable Diffusion 进行受控生成和微调
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# 问题与解答
_问：什么是 ControlNet，它如何在 Stable Diffusion 中使用？_

答：ControlNet 是一种深度神经网络，与 Stable Diffusion 等扩散模型协同工作。在训练过程中，ControlNet 会从给定的输入集中学习特定任务，例如边缘检测或深度映射。它用于训练各种控制，这些控制使用相对少量的训练数据来改进基于图像的生成式任务。

_问：DreamBooth 如何促进 Stable Diffusion 的微调？_

答：DreamBooth 允许使用少量的图像来微调 Stable Diffusion 模型。它支持属性修改和配饰，允许你修改输入图像的特定方面，例如颜色，或保留主体但使用配饰修改图像。

_问：什么是文本反转，它与微调有什么关系？_

答：文本反转是一种轻量级的微调技术，用于使用少量的图像来个性化基于图像的生成式模型。它的工作原理是学习一个新文本标记的标记嵌入，该标记代表一个概念，同时保持 Stable Diffusion 模型的其余部分冻结。

_问：人类与 RLHF 的对齐如何增强 Stable Diffusion 模型？_

答：人类与基于人类反馈的强化学习 (RLHF) 的对齐可以微调扩散模型，以改进图像可压缩性、美学质量和提示-图像对齐等方面。RLHF 使多模态模型能够生成更有益、更诚实且更无害的内容。

_问：PEFT-LoRA 技术如何帮助微调 Stable Diffusion 模型？"

答：PEFT-LoRA（使用低秩自适应的参数高效微调）可用于微调 Stable Diffusion 模型的交叉注意力层。它提供了一种高效的方式来调整这些模型，而无需进行大规模的重新训练。

_问：微调 Stable Diffusion 模型有哪些好处？_

答：微调 Stable Diffusion 模型允许定制图像生成，以包括原始训练数据语料库中未捕获的图像数据。这可以包括任何图像数据，例如人、宠物或徽标的图像，从而能够生成包含基础模型未知主体的逼真图像。

_问：使用 Stable Diffusion 模型进行微调与其他生成式人工智能模型有何不同？_

答：Stable Diffusion 等扩散模型的微调技术与用于基于 Transformer 的大型语言模型 (LLM) 的技术类似。这些技术允许定制图像生成，以包含特定数据集或主题，类似于 LLM 如何被微调以与特定的文本数据或风格保持一致。

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
