# HuggingChat 与 OpenAI 集成

这个仓库包含的代码可以将 [Hugging Chat UI](https://github.com/huggingface/chat-ui) 与 OpenAI API 集成。提供的 Python FastAPI 代码允许你使用 OpenAI GPT-3.5-turbo 模型生成聊天补全，并将结果实时流式传输到 Hugging Chat UI。

## 功能

- 将 Hugging Chat UI 与 OpenAI API 集成，用于聊天补全。
- 实时将聊天补全结果流式传输到 UI。
- 可自定义的生成聊天补全的参数。
- 使用 FastAPI 进行轻松的设置和部署。

## 要求

要在本仓库中运行代码，你需要以下内容：

- Python 3.7 或更高版本
- FastAPI
- Pydantic
- OpenAI Python SDK
- asyncio

你还需要一个有效的 OpenAI API 密钥来验证你的请求。

## 安装

1. 克隆仓库
2. 安装所需的依赖项：
`pip install -r requirements.txt`
3. 设置你的 OpenAI API 密钥：

    注册 OpenAI 帐户并获取 API 密钥。

    将你的 API 密钥设置为 OPENAI_API_KEY 环境变量。

## 使用

1. 通过运行以下命令启动 FastAPI 服务器：

    ```uvicorn server:app --reload```

    服务器将默认在 http://localhost:8000 上运行。
2. 更新 HuggingChat UI 安装中的 `.env.local` 文件，以包含：

    ```python
    MODELS=`[
    {
        "name": "ChatGPT 3.5 Model",
        "endpoints": [{"url": "http://127.0.0.1:8000/generate_stream"}],
        "userMessageToken": "User: ",
        "assistantMessageToken": "Assistant: ",
        "messageEndToken": "\n",
        "preprompt": "You are a helpful assistant.",
        "parameters": {
        "temperature": 0.9,
        "max_new_tokens": 50,
        "truncate": 1000
        }
    }
    ]`
    ```

## 测试

向 http://localhost:8000/generate_stream 发送一个 POST 请求，并使用以下 JSON 负载：

```json
{
"inputs": "User input message",
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 100
    }
}
```
OpenAI GPT-3.5-turbo 模型生成的聊天补全将实时流式传输到 UI。

1. 使用 Curl，你可以通过以下方式进行测试：
    ```bash
    curl 127.0.0.1:8000/generate_stream \
        -X POST \
        -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17}}' \
        -H 'Content-Type: application/json'
    ```

## 许可证

此项目根据 GPL3 许可证授权。

**改编自：**  https://github.com/gururise/openai_text_generation_inference_server

这个项目是基于 [gururise/openai_text_generation_inference_server](https://github.com/gururise/openai_text_generation_inference_server) 项目进行改编的。
