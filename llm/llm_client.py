"""
统一的 LLM 客户端封装
支持多种模型提供商：OpenAI、豆包、通义千问、文心一言等
"""

import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# 确保从项目根目录加载 .env 文件
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()  # 回退到默认行为


class LLMClient:
    """统一的 LLM 客户端，支持多种模型提供商"""
    
    def __init__(
        self,
        provider: str = "doubao",  # 默认使用豆包
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        初始化 LLM 客户端
        
        Args:
            provider: 提供商名称 (openai, doubao, qwen, ernie, zhipu)
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL（用于自定义端点）
            temperature: 温度参数
            max_tokens: 最大 token 数
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 设置默认模型和配置
        self._setup_provider_config(model_name, api_key, base_url)
        
        # 初始化客户端
        self._init_client()
    
    def _setup_provider_config(
        self,
        model_name: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str]
    ):
        """设置提供商配置"""
        provider_configs = {
            "openai": {
                "model": model_name or "gpt-3.5-turbo",
                "api_key": api_key or os.getenv("OPENAI_API_KEY"),
                "base_url": base_url or "https://api.openai.com/v1"
            },
            "doubao": {
                "model": model_name or "doubao-1-5-lite-32k-250115",  # 豆包模型
                "api_key": api_key or os.getenv("DOUBAO_API_KEY"),
                "base_url": base_url or "https://ark.cn-beijing.volces.com/api/v3"
            },
            "qwen": {
                "model": model_name or "qwen-turbo",
                "api_key": api_key or os.getenv("DASHSCOPE_API_KEY"),
                "base_url": base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            "ernie": {
                "model": model_name or "ernie-bot-turbo",
                "api_key": api_key or os.getenv("ERNIE_API_KEY"),
                "base_url": base_url or "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
            },
            "zhipu": {
                "model": model_name or "glm-4",
                "api_key": api_key or os.getenv("ZHIPU_API_KEY"),
                "base_url": base_url or "https://open.bigmodel.cn/api/paas/v4"
            }
        }
        
        if self.provider not in provider_configs:
            raise ValueError(
                f"不支持的提供商: {self.provider}。"
                f"支持的提供商: {', '.join(provider_configs.keys())}"
            )
        
        config = provider_configs[self.provider]
        self.model_name = config["model"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        
        if not self.api_key:
            raise ValueError(
                f"请设置 {self.provider.upper()}_API_KEY 环境变量或传入 api_key 参数"
            )
    
    def _init_client(self):
        """初始化客户端"""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.provider == "doubao":
            # 豆包使用 OpenAI 兼容的 API
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.provider == "qwen":
            # 通义千问使用 OpenAI 兼容的 API
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.provider == "zhipu":
            # 智谱 GLM 使用 OpenAI 兼容的 API
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.provider == "ernie":
            # 文心一言需要特殊处理
            import requests
            self.client = None  # 文心一言使用 requests
            self._ernie_api_key = self.api_key
        else:
            raise ValueError(f"不支持的提供商: {self.provider}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            
        Returns:
            模型返回的文本
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        if self.provider == "ernie":
            return self._chat_ernie(messages, temperature, max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM API 调用失败: {e}")
    
    def _chat_ernie(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """文心一言的特殊处理"""
        import requests
        import json
        
        # 文心一言需要将消息转换为特定格式
        conversation = []
        for msg in messages:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        url = f"{self.base_url}/{self.model_name}"
        headers = {
            "Content-Type": "application/json"
        }
        
        # 文心一言的 API Key 格式特殊
        api_key_parts = self._ernie_api_key.split(":")
        if len(api_key_parts) == 2:
            headers["Authorization"] = f"Bearer {api_key_parts[0]}"
            params = {
                "access_token": api_key_parts[1]
            }
        else:
            params = {
                "access_token": self._ernie_api_key
            }
        
        payload = {
            "messages": conversation,
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "result" in result:
            return result["result"]
        else:
            raise Exception(f"文心一言 API 返回错误: {result}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        生成文本（简化接口）
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens)


# 便捷函数
def get_llm_client(
    provider: str = None,
    **kwargs
) -> LLMClient:
    """
    获取 LLM 客户端的便捷函数
    
    Args:
        provider: 提供商名称，如果为 None 则从环境变量 LLM_PROVIDER 读取
        **kwargs: 其他参数传递给 LLMClient
        
    Returns:
        LLMClient 实例
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "doubao")
    
    return LLMClient(provider=provider, **kwargs)


if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("LLM 客户端测试")
    print("=" * 60)
    
    # 测试豆包（需要设置 DOUBAO_API_KEY）
    try:
        client = get_llm_client(provider="doubao")
        response = client.generate("你好，请介绍一下你自己")
        print(f"\n豆包回复: {response}")
    except Exception as e:
        print(f"\n豆包测试失败: {e}")
        print("提示: 请设置 DOUBAO_API_KEY 环境变量")
    
    # 测试 OpenAI（需要设置 OPENAI_API_KEY）
    try:
        client = get_llm_client(provider="openai")
        response = client.generate("Hello, please introduce yourself")
        print(f"\nOpenAI 回复: {response}")
    except Exception as e:
        print(f"\nOpenAI 测试失败: {e}")
        print("提示: 请设置 OPENAI_API_KEY 环境变量")

