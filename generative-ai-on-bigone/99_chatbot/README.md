## Chat UI

![Chat UI 仓库缩略图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chatui-websearch.png)

使用开源模型（例如 OpenAssistant 或 Llama）的聊天界面。它是一个 SvelteKit 应用，为 [hf.co/chat 上的 HuggingChat 应用](https://huggingface.co/chat) 提供支持。

0. [无需设置部署](#no-setup-deploy)
1. [设置](#setup)
2. [启动](#launch)
3. [网页搜索](#web-search)
4. [文本嵌入模型](#text-embedding-models)
5. [额外参数](#extra-parameters)
6. [部署到 HF Space](#deploying-to-a-hf-space)
7. [构建](#building)

## 无需设置部署

如果你不想自己配置、设置和启动自己的 Chat UI，可以使用此选项作为快速部署的替代方案。

你可以在 [Hugging Face Spaces](https://huggingface.co/spaces) 上部署你自己的自定义 Chat UI 实例，并使用任何支持的 [LLM](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)。为此，请使用 [此处提供的 chat-ui 模板](https://huggingface.co/new-space?template=huggingchat/chat-ui-template)。

在 [Space secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables) 中设置 `HF_TOKEN` 以部署具有门控访问权限的模型或私有仓库中的模型。它也与 [Inference for PROs](https://huggingface.co/blog/inference-pro) 上的精选强大的模型列表兼容，这些模型具有更高的速率限制。请确保先在你的 [用户访问令牌设置](https://huggingface.co/settings/tokens) 中创建你的个人令牌。

阅读完整的教程 [此处](https://huggingface.co/docs/hub/spaces-sdks-docker-chatui#chatui-on-spaces)。

## 设置

Chat UI 的默认配置存储在 `.env` 文件中。你需要覆盖一些值才能使 Chat UI 在本地运行。这在 `.env.local` 中完成。

首先在仓库的根目录中创建一个 `.env.local` 文件。要使 Chat UI 在本地运行，你需要的最少配置如下：

```env
MONGODB_URL=<你的 MongoDB 实例的 URL>
HF_TOKEN=<你的访问令牌>
```

### 数据库

聊天历史记录存储在 MongoDB 实例中，要使 Chat UI 正常运行，需要有一个可用的 DB 实例。

你可以使用本地 MongoDB 实例。最简单的方法是使用 docker 启动一个：

```bash
docker run -d -p 27017:27017 --name mongo-chatui mongo:latest
```

在这种情况下，你的 DB 的 url 将是 `MONGODB_URL=mongodb://localhost:27017`。

或者，你可以为此使用 [免费的 MongoDB Atlas](https://www.mongodb.com/pricing) 实例，Chat UI 应该可以轻松地适应他们的免费层级。之后，你可以在 `.env.local` 中设置 `MONGODB_URL` 变量以匹配你的实例。

### Hugging Face 访问令牌

如果你使用远程推理端点，你需要一个 Hugging Face 访问令牌才能在本地运行 Chat UI。你可以在 [你的 Hugging Face 个人资料](https://huggingface.co/settings/tokens) 中获取一个。

## 启动

完成 `.env.local` 文件后，你可以使用以下命令在本地运行 Chat UI：

```bash
npm install
npm run dev
```

## 网页搜索

Chat UI 具有强大的网页搜索功能。它的工作原理是：

1. 从用户提示中生成一个合适的搜索查询。
2. 执行网页搜索并从网页中提取内容。
3. 使用文本嵌入模型从文本中创建嵌入。
4. 从这些嵌入中，使用向量相似性搜索找到最接近用户查询的嵌入。具体来说，我们使用 `内积` 距离。
5. 获取与这些最接近嵌入相对应的文本，并执行 [检索增强生成](https://huggingface.co/papers/2005.11401)（即通过添加这些文本来扩展用户提示，以便 LLM 可以使用这些信息）。

## 文本嵌入模型

默认情况下（为了向后兼容性），当 `TEXT_EMBEDDING_MODELS` 环境变量未定义时，[transformers.js](https://huggingface.co/docs/transformers.js) 嵌入模型将用于嵌入任务，具体来说，是 [Xenova/gte-small](https://huggingface.co/Xenova/gte-small) 模型。

你可以在 `.env.local` 文件中设置 `TEXT_EMBEDDING_MODELS` 来自定义嵌入模型。例如：

```env
TEXT_EMBEDDING_MODELS = `[
  {
    "name": "Xenova/gte-small",
    "displayName": "Xenova/gte-small",
    "description": "本地运行的嵌入",
    "chunkCharLength": 512,
    "endpoints": [
      {"type": "transformersjs"}
    ]
  },
  {
    "name": "intfloat/e5-base-v2",
    "displayName": "intfloat/e5-base-v2",
    "description": "托管的嵌入模型",
    "chunkCharLength": 768,
    "preQuery": "query: ", # See https://huggingface.co/intfloat/e5-base-v2#faq
    "prePassage": "passage: ", # See https://huggingface.co/intfloat/e5-base-v2#faq
    "endpoints": [
      {
        "type": "tei",
        "url": "http://127.0.0.1:8080/",
        "authorization": "TOKEN_TYPE TOKEN" // 可选的授权字段。示例："Basic VVNFUjpQQVNT"
      }
    ]
  }
]`
```

必需字段是 `name`、`chunkCharLength` 和 `endpoints`。
支持的文本嵌入后端是：[`transformers.js`](https://huggingface.co/docs/transformers.js) 和 [`TEI`](https://github.com/huggingface/text-embeddings-inference)。`transformers.js` 模型作为 `chat-ui` 的一部分在本地运行，而 `TEI` 模型在不同的环境中运行，并通过 API 端点访问。

当 `.env.local` 文件中提供多个嵌入模型时，第一个模型将默认使用，而其他模型只会在配置了 `embeddingModel` 为模型名称的 LLM 上使用。

## 额外参数

### OpenID 连接

登录功能默认情况下是禁用的，用户会根据其浏览器分配一个唯一的 ID。但如果你想使用 OpenID 来验证你的用户，你可以在你的 `.env.local` 文件中添加以下内容：

```env
OPENID_CONFIG=`{
  PROVIDER_URL: "<你的 OIDC 发行者>",
  CLIENT_ID: "<你的 OIDC 客户端 ID>",
  CLIENT_SECRET: "<你的 OIDC 客户端密钥>",
  SCOPES: "openid profile",
  TOLERANCE: // 可选
  RESOURCE: // 可选
}`
```

这些变量将为用户启用 OpenID 登录模态。

### 主题

你可以使用一些环境变量来自定义 chat-ui 的外观和感觉。默认情况下，它们是：

```env
PUBLIC_APP_NAME=ChatUI
PUBLIC_APP_ASSETS=chatui
PUBLIC_APP_COLOR=blue
PUBLIC_APP_DESCRIPTION="Making the community's best AI chat models available to everyone."
PUBLIC_APP_DATA_SHARING=
PUBLIC_APP_DISCLAIMER=
```

- `PUBLIC_APP_NAME` 在整个应用程序中用作标题的名称。
- `PUBLIC_APP_ASSETS` 用于在 `static/$PUBLIC_APP_ASSETS` 中查找徽标和收藏夹图标，当前选项是 `chatui` 和 `huggingchat`。
- `PUBLIC_APP_COLOR` 可以是任何 [tailwind 颜色](https://tailwindcss.com/docs/customizing-colors#default-color-palette)。
- `PUBLIC_APP_DATA_SHARING` 可以设置为 1，以便在用户设置中添加一个切换按钮，让你的用户选择是否与模型创建者共享数据。
- `PUBLIC_APP_DISCLAIMER` 如果设置为 1，我们将在登录时显示有关生成输出的免责声明。

### 网页搜索配置

你可以通过添加 `YDC_API_KEY` ([docs.you.com](https://docs.you.com)) 或 `SERPER_API_KEY` ([serper.dev](https://serper.dev/)) 或 `SERPAPI_KEY` ([serpapi.com](https://serpapi.com/)) 或 `SERPSTACK_API_KEY` ([serpstack.com](https://serpstack.com/)) 到你的 `.env.local` 中，通过 API 启用网页搜索。

你也可以简单地通过在你的 `.env.local` 中设置 `USE_LOCAL_WEBSEARCH=true` 来启用本地 google 网页搜索，或者通过将查询 URL 添加到 `SEARXNG_QUERY_URL` 来指定 SearXNG 实例。

### 自定义模型

你可以通过更新 `.env.local` 中的 `MODELS` 变量来自定义传递给模型的参数，甚至使用新的模型。默认模型可以在 `.env` 中找到，如下所示：

```env
MODELS=`[
  {
    "name": "mistralai/Mistral-7B-Instruct-v0.2",
    "displayName": "mistralai/Mistral-7B-Instruct-v0.2",
    "description": "Mistral 7B 是一个新的 Apache 2.0 模型，由 Mistral AI 发布，在基准测试中优于 Llama2 13B。",
    "websiteUrl": "https://mistral.ai/news/announcing-mistral-7b/",
    "preprompt": "",
    "chatPromptTemplate" : "<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}}{{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s>{{/ifAssistant}}{{/each}}",
    "parameters": {
      "temperature": 0.3,
      "top_p": 0.95,
      "repetition_penalty": 1.2,
      "top_k": 50,
      "truncate": 3072,
      "max_new_tokens": 1024,
      "stop": ["</s>"]
    },
    "promptExamples": [
      {
        "title": "从项目符号列表中编写电子邮件",
        "prompt": "作为餐厅老板，写一封专业的电子邮件给供应商，要求每周获得以下产品：\n\n- 葡萄酒 (x10)\n- 鸡蛋 (x24)\n- 面包 (x12)"
      }, {
        "title": "编写一个贪吃蛇游戏",
        "prompt": "用 python 编写一个基本的贪吃蛇游戏，为每一步提供解释。"
      }, {
        "title": "协助完成任务",
        "prompt": "如何制作美味的柠檬芝士蛋糕？"
      }
    ]
  }
]`

```

你可以更改参数等内容，或自定义 preprompt 以更好地满足你的需求。你还可以通过向数组中添加更多对象来添加更多模型，例如使用不同的 preprompt。

#### chatPromptTemplate

当向模型查询聊天响应时，将使用 `chatPromptTemplate` 模板。`messages` 是聊天消息的数组，其格式为 `[{ content: string }, ...]`。要识别消息是用户消息还是助手消息，可以使用 `ifUser` 和 `ifAssistant` 块助手。

以下是默认的 `chatPromptTemplate`，尽管为了可读性添加了换行符和缩进。你可以在此处找到用于 HuggingChat 生产环境的提示 [此处](https://github.com/huggingface/chat-ui/blob/main/PROMPTS.md)。

```prompt
{{preprompt}}
{{#each messages}}
  {{#ifUser}}{{@root.userMessageToken}}{{content}}{{@root.userMessageEndToken}}{{/ifUser}}
  {{#ifAssistant}}{{@root.assistantMessageToken}}{{content}}{{@root.assistantMessageEndToken}}{{/ifAssistant}}
{{/each}}
{{assistantMessageToken}}
```

#### 多模态模型

我们目前只支持 IDEFICS 作为多模态模型，它托管在 TGI 上。你可以使用以下配置来启用它（如果你拥有 PRO HF API 令牌）：

```env
    {
      "name": "HuggingFaceM4/idefics-80b-instruct",
      "multimodal" : true,
      "description": "IDEFICS 是 Hugging Face 的新多模态模型。",
      "preprompt": "",
      "chatPromptTemplate" : "{{#each messages}}{{#ifUser}}User: {{content}}{{/ifUser}}<end_of_utterance>\nAssistant: {{#ifAssistant}}{{content}}\n{{/ifAssistant}}{{/each}}",
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 12,
        "truncate": 1000,
        "max_new_tokens": 1024,
        "stop": ["<end_of_utterance>", "User:", "\nUser:"]
      }
    }
```

#### 使用自定义端点运行你自己的模型

如果你想在本地运行你自己的模型，而不是在 Hugging Face 推理 API 上运行模型，你可以这样做。

一个不错的选择是使用 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 端点。例如，在官方的 [Chat UI Spaces Docker 模板](https://huggingface.co/new-space?template=huggingchat/chat-ui-template) 中，这个应用和一个 text-generation-inference 服务器都在同一个容器中运行。

为此，你可以在 `.env.local` 中的 `MODELS` 变量中添加你自己的端点，方法是为 `MODELS` 中的每个模型添加一个 `"endpoints"` 键。

```env
{
// 这里还有模型配置
"endpoints": [{
  "type" : "tgi",
  "url": "https://HOST:PORT",
  }]
}
```

如果 `endpoints` 未指定，ChatUI 将使用模型名称在托管的 Hugging Face 推理 API 上查找模型。

##### 兼容 OpenAI API 的模型

Chat UI 可以与任何支持 OpenAI API 兼容性的 API 服务器一起使用，例如 [text-generation-webui](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)、[LocalAI](https://github.com/go-skynet/LocalAI)、[FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)、[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 和 [ialacol](https://github.com/chenhunghan/ialacol)。

以下示例配置使 Chat UI 可以与 [text-generation-webui](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai) 一起使用，`endpoint.baseUrl` 是兼容 OpenAI API 的服务器的 url，它覆盖了 OpenAI 实例使用的 baseUrl。`endpoint.completion` 决定使用哪个端点，默认是 `chat_completions`，它使用 `v1/chat/completions`，更改为 `endpoint.completion` 为 `completions` 以使用 `v1/completions` 端点。

```
MODELS=`[
  {
    "name": "text-generation-webui",
    "id": "text-generation-webui",
    "parameters": {
      "temperature": 0.9,
      "top_p": 0.95,
      "repetition_penalty": 1.2,
      "top_k": 50,
      "truncate": 1000,
      "max_new_tokens": 1024,
      "stop": []
    },
    "endpoints": [{
      "type" : "openai",
      "baseURL": "http://localhost:8000/v1"
    }]
  }
]`

```

`openai` 类型包括官方的 OpenAI 模型。例如，你可以添加 GPT4/GPT3.5 作为 "openai" 模型：

```
OPENAI_API_KEY=#你的 openai api 密钥
MODELS=`[{
      "name": "gpt-4",
      "displayName": "GPT 4",
      "endpoints" : [{
        "type": "openai"
      }]
},
      {
      "name": "gpt-3.5-turbo",
      "displayName": "GPT 3.5 Turbo",
      "endpoints" : [{
        "type": "openai"
      }]
}]`
```

你也可以使用任何提供兼容 OpenAI API 端点的模型提供商。例如，你可以自托管 [Portkey](https://github.com/Portkey-AI/gateway) 网关，并尝试使用 Azure OpenAI 提供的 Claude 或 GPT。来自 Anthropic 的 Claude 示例：

```
MODELS=`[{
  "name": "claude-2.1",
  "displayName": "Claude 2.1",
  "description": "Anthropic 由前 OpenAI 研究人员创立...",
  "parameters": {
      "temperature": 0.5,
      "max_new_tokens": 4096,
  },
  "endpoints": [
      {
          "type": "openai",
          "baseURL": "https://gateway.example.com/v1",
          "defaultHeaders": {
              "x-portkey-config": '{"provider":"anthropic","api_key":"sk-ant-abc...xyz"}'
          }
      }
  ]
}]`
```

部署在 Azure OpenAI 上的 GPT 4 示例：

```
MODELS=`[{
  "id": "gpt-4-1106-preview",
  "name": "gpt-4-1106-preview",
  "displayName": "gpt-4-1106-preview",
  "parameters": {
      "temperature": 0.5,
      "max_new_tokens": 4096,
  },
  "endpoints": [
      {
          "type": "openai",
          "baseURL": "https://{resource-name}.openai.azure.com/openai/deployments/{deployment-id}",
          "defaultHeaders": {
              "api-key": "{api-key}"
          },
          "defaultQuery": {
              "api-version": "2023-05-15"
          }
      }
  ]
}]`
```

或者尝试来自 [Deepinfra](https://deepinfra.com/mistralai/Mistral-7B-Instruct-v0.1/api?example=openai-http) 的 Mistral：

> 注意，apiKey 可以针对每个端点进行自定义设置，也可以使用 `OPENAI_API_KEY` 变量全局设置。

```
MODELS=`[{
  "name": "mistral-7b",
  "displayName": "Mistral 7B",
  "description": "一个 7B 密集 Transformer，快速部署且易于定制。体积小，但功能强大，适用于各种用例。支持英语和代码，以及 8k 上下文窗口。",
  "parameters": {
      "temperature": 0.5,
      "max_new_tokens": 4096,
  },
  "endpoints": [
      {
          "type": "openai",
          "baseURL": "https://api.deepinfra.com/v1/openai",
          "apiKey": "abc...xyz"
      }
  ]
}]`
```

##### Llama.cpp API 服务器

chat-ui 也直接支持 llama.cpp API 服务器，无需适配器。你可以使用 `llamacpp` 端点类型来实现。

如果你想使用 llama.cpp 运行 chat-ui，你可以执行以下操作，以 Zephyr 作为示例模型：

1. 从中心获取 [权重](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/tree/main)
2. 使用以下命令运行服务器：`./server -m models/zephyr-7b-beta.Q4_K_M.gguf -c 2048 -np 3`
3. 将以下内容添加到你的 `.env.local` 中：

```env
MODELS=`[
  {
      "name": "Local Zephyr",
      "chatPromptTemplate": "<|system|>\n{{preprompt}}</s>\n{{#each messages}}{{#ifUser}}<|user|>\n{{content}}</s>\n<|assistant|>\n{{/ifUser}}{{#ifAssistant}}{{content}}</s>\n{{/ifAssistant}}{{/each}}",
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "truncate": 1000,
        "max_new_tokens": 2048,
        "stop": ["</s>"]
      },
      "endpoints": [
        {
         "url": "http://127.0.0.1:8080",
         "type": "llamacpp"
        }
      ]
  }
]`
```

使用 `npm run dev` 启动 chat-ui，你应该能够在本地与 Zephyr 聊天。

#### Ollama

我们还支持 Ollama 推理服务器。使用以下命令启动一个模型：

```cli
ollama run mistral
```

然后像这样指定端点：

```env
MODELS=`[
  {
      "name": "Ollama Mistral",
      "chatPromptTemplate": "<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}",
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "truncate": 3072,
        "max_new_tokens": 1024,
        "stop": ["</s>"]
      },
      "endpoints": [
        {
         "type": "ollama",
         "url" : "http://127.0.0.1:11434",
         "ollamaName" : "mistral"
        }
      ]
  }
]`
```

#### Amazon

你还可以将你的 Amazon SageMaker 实例指定为 chat-ui 的端点。配置如下所示：

```env
"endpoints": [
    {
      "type" : "aws",
      "service" : "sagemaker"
      "url": "",
      "accessKey": "",
      "secretKey" : "",
      "sessionToken": "",
      "region": "",

      "weight": 1
    }
]
```

你也可以设置 `"service" : "lambda"` 以使用 lambda 实例。

你可以在你的 AWS 用户的程序访问权限下获取 `accessKey` 和 `secretKey`。

### 自定义端点授权

#### Basic 和 Bearer

自定义端点可能需要授权，具体取决于你如何配置它们。身份验证通常使用 `Basic` 或 `Bearer` 设置。

对于 `Basic`，我们需要生成用户名和密码的 base64 编码。

`echo -n "USER:PASS" | base64`

> VVNFUjpQQVNT

对于 `Bearer`，你可以使用令牌，可以从 [这里](https://huggingface.co/settings/tokens) 获取。

然后，你可以将生成的 information 和 `authorization` 参数添加到你的 `.env.local` 中。

```env
"endpoints": [
  {
    "url": "https://HOST:PORT",
    "authorization": "Basic VVNFUjpQQVNT",
  }
]
```

请注意，如果 `HF_TOKEN` 也设置了或不为空，它将优先使用。

#### 托管在多个自定义端点上的模型

如果托管的模型将在多个服务器/实例上可用，请将 `weight` 参数添加到你的 `.env.local` 中。`weight` 将用于确定请求特定端点的概率。

```env
"endpoints": [
  {
    "url": "https://HOST:PORT",
    "weight": 1
  },
  {
    "url": "https://HOST:PORT",
    "weight": 2
  }
  ...
]
```

#### 客户端证书身份验证 (mTLS)

自定义端点可能需要客户端证书身份验证，具体取决于你如何配置它们。要启用 Chat UI 和自定义端点之间的 mTLS，你需要将 `USE_CLIENT_CERTIFICATE` 设置为 `true`，并将 `CERT_PATH` 和 `KEY_PATH` 参数添加到你的 `.env.local` 中。这些参数应该指向证书和密钥文件在你的本地机器上的位置。证书和密钥文件应该使用 PEM 格式。密钥文件可以使用密码进行加密，在这种情况下，你还需要将 `CLIENT_KEY_PASSWORD` 参数添加到你的 `.env.local` 中。

如果你使用的是由私有 CA 签名的证书，你还需要将 `CA_PATH` 参数添加到你的 `.env.local` 中。此参数应该指向 CA 证书文件在你的本地机器上的位置。

如果你使用的是自签名证书（例如，用于测试或开发目的），你可以在你的 `.env.local` 中将 `REJECT_UNAUTHORIZED` 参数设置为 `false`。这将禁用证书验证，并允许 Chat UI 连接到你的自定义端点。

#### 特定嵌入模型

模型可以使用 `.env.local` 中定义的任何嵌入模型（目前在网页搜索时使用），默认情况下它将使用第一个嵌入模型，但可以使用 `embeddingModel` 字段进行更改：

```env
TEXT_EMBEDDING_MODELS = `[
  {
    "name": "Xenova/gte-small",
    "chunkCharLength": 512,
    "endpoints": [
      {"type": "transformersjs"}
    ]
  },
  {
    "name": "intfloat/e5-base-v2",
    "chunkCharLength": 768,
    "endpoints": [
      {"type": "tei", "url": "http://127.0.0.1:8080/", "authorization": "Basic VVNFUjpQQVNT"},
      {"type": "tei", "url": "http://127.0.0.1:8081/"}
    ]
  }
]`

MODELS=`[
  {
      "name": "Ollama Mistral",
      "chatPromptTemplate": "...",
      "embeddingModel": "intfloat/e5-base-v2"
      "parameters": {
        ...
      },
      "endpoints": [
        ...
      ]
  }
]`
```

## 部署到 HF Space

创建一个包含 `.env.local` 内容的 `DOTENV_LOCAL` 密钥到你的 HF 空间，它们将在你运行时自动被拾取。

## 构建

要创建应用程序的生产版本，请执行以下操作：

```bash
npm run build
```

你可以使用 `npm run preview` 预览生产构建。

> 要部署你的应用程序，你可能需要为你的目标环境安装一个 [适配器](https://kit.svelte.dev/docs/adapters)。

## HuggingChat 的配置更改

HuggingChat 的配置文件存储在仓库根目录下的 `.env.template` 文件中。它是用于使用我们的 CI/CD 管道生成实际的 `.env.local` 文件的唯一真实来源。有关更多详细信息，请参阅 [updateProdEnv](https://github.com/huggingface/chat-ui/blob/cdb33a9583f5339ade724db615347393ef48f5cd/scripts/updateProdEnv.ts)。

> [!TIP]
> 如果你想更改 HuggingChat 的模型配置，你应该针对 `.env.template` 进行更改。

除了上面的 `.env.template` 之外，我们目前还使用以下密钥来部署 HuggingChat：

- `MONGODB_URL`
- `HF_TOKEN`
- `OPENID_CONFIG`
- `SERPER_API_KEY`

它们在仓库中定义为密钥。

### 在本地测试配置更改

你可以通过首先创建一个包含上述密钥的 `.env.SECRET_CONFIG` 文件来在本地测试配置更改。然后，你可以运行以下命令来生成 `.env.local` 文件：

```bash
npm run updateLocalEnv
```

这将用将在生产环境中使用的文件替换你的 `.env.local` 文件（简单地获取 `.env.template + .env.SECRET_CONFIG`）。
