# Stable Diffusion XL (SDXL) 的 DreamBooth 训练示例

[DreamBooth](https://arxiv.org/abs/2208.12242) 是一种个性化文本到图像模型（如 Stable Diffusion）的方法，只需使用主题的几张（3-5 张）图像即可。

`train_dreambooth_lora_sdxl.py` 脚本展示了如何实现训练过程并将其适应于 [Stable Diffusion XL](https://huggingface.co/papers/2307.01952)。

> 💡 **注意**: 目前，我们只允许通过 LoRA 对 SDXL UNet 进行 DreamBooth 微调。LoRA 是一种参数高效的微调技术，由 *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* 在 [LoRA: 大型语言模型的低秩自适应](https://arxiv.org/abs/2106.09685) 中提出。

## 使用 PyTorch 在本地运行

### 安装依赖项

在运行脚本之前，请确保安装库的训练依赖项：

**重要**

为了确保你能够成功运行最新版本的示例脚本，我们强烈建议你 **从源代码安装** 并保持安装更新，因为我们经常更新示例脚本并安装一些示例特定的要求。为此，请在新虚拟环境中执行以下步骤：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

然后进入 `examples/dreambooth` 文件夹并运行
```bash
pip install -r requirements_sdxl.txt
```

并使用以下命令初始化一个 [🤗Accelerate](https://github.com/huggingface/accelerate/) 环境：

```bash
accelerate config
```

或者，对于不回答有关环境问题的默认加速配置

```bash
accelerate config default
```

或者，如果你的环境不支持交互式 shell（例如，笔记本）

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

在运行 `accelerate config` 时，如果我们将 torch 编译模式设置为 True，则可以显着提高速度。

### 狗玩具示例

现在让我们获取数据集。在本示例中，我们将使用一些狗的图像：https://huggingface.co/datasets/diffusers/dog-example。

让我们首先将其下载到本地：

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

这也将允许我们将训练好的 LoRA 参数推送到 Hugging Face Hub 平台。

现在，我们可以使用以下命令启动训练：

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

为了更好地跟踪我们的训练实验，我们在上面的命令中使用了以下标志：

* `report_to="wandb"` 将确保在 Weights and Biases 上跟踪训练运行。要使用它，请确保使用 `pip install wandb` 安装 `wandb`。
* `validation_prompt` 和 `validation_epochs` 允许脚本执行一些验证推理运行。这使我们能够定性地检查训练是否按预期进行。

我们的实验是在单个 40GB A100 GPU 上进行的。

### 使用小于 16GB VRAM 的狗玩具示例

通过利用 [`gradient_checkpointing`](https://pytorch.org/docs/stable/checkpoint.html)（在 Diffusers 中得到原生支持）、[`xformers`](https://github.com/facebookresearch/xformers) 和 [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) 库，你可以通过在你的 accelerate launch 命令中添加以下标志来训练使用小于 16GB VRAM 的 SDXL LoRA：

```diff
+  --enable_xformers_memory_efficient_attention \
+  --gradient_checkpointing \
+  --use_8bit_adam \
+  --mixed_precision="fp16" \
```

并确保你已安装以下库：

```
bitsandbytes>=0.40.0
xformers>=0.0.20
```

### 推理

训练完成后，我们可以执行以下推理：

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog.png")
```

我们可以使用 [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) 进一步优化输出：

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

# 加载基础管道并将 LoRA 参数加载到其中。
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)

# 加载细化器。
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

prompt = "A picture of a sks dog in a bucket"
generator = torch.Generator("cuda").manual_seed(0)

# 运行推理。
image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
image.save("refined_sks_dog.png")
```

以下是使用和不使用 Refiner 管道的输出的并排比较：

| 不使用 Refiner | 使用 Refiner |
|---|---|
| ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/sks_dog.png) | ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_sks_dog.png) |

### 使用文本编码器进行训练

除了 UNet 之外，还支持对文本编码器进行 LoRA 微调。为此，只需在启动训练时指定 `--train_text_encoder`。请记住以下几点：

* SDXL 具有两个文本编码器。因此，我们使用 LoRA 对两者进行微调。
* 在不微调文本编码器的情况下，我们始终预先计算文本嵌入以节省内存。

### 使用更好的 VAE

SDXL 的 VAE 存在数值不稳定性问题。这就是我们公开 CLI 参数 `--pretrained_vae_model_name_or_path` 的原因，该参数允许你指定更好 VAE 的位置（例如 [这个](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)）。

## 注意

在我们的实验中，我们发现 SDXL 在没有进行大量超参数调整的情况下就能产生良好的初始结果。例如，在不微调文本编码器且不使用先验保留的情况下，我们观察到了不错的结果。我们没有进行进一步的超参数调整实验，但我们鼓励社区进一步探索这一途径，并将他们的结果与我们分享 🤗

## 结果

你可以通过查看此链接来探索我们的一些内部实验的结果：[https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl](https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl)。具体来说，我们在以下数据集上使用相同的脚本和完全相同的超参数：

* [狗](https://huggingface.co/datasets/diffusers/dog-example)
* [星巴克 logo](https://huggingface.co/datasets/diffusers/starbucks-example)
* [土豆先生](https://huggingface.co/datasets/diffusers/potato-head-example)
* [Keramer 的脸](https://huggingface.co/datasets/diffusers/keramer-face-example)

## 在免费层级 Colab 笔记本上运行

查看 [这个笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb)。 
