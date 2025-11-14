"""
Structured output using DSPy and Guidance.
Day 11-13: 结构化输出与抽取
"""

from typing import Dict, List, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()


class StructuredOutputDemo:
    """结构化输出示例：使用 DSPy 和 Guidance"""
    
    def __init__(self):
        """初始化结构化输出模块"""
        pass
    
    def extract_with_dspy(
        self,
        text: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 DSPy 进行结构化抽取
        
        Args:
            text: 输入文本
            schema: 输出 schema 定义
            
        Returns:
            结构化数据
        """
        try:
            import dspy
            
            # 配置 LLM（需要先设置）
            # dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))
            
            # 定义 Signature
            class ExtractSignature(dspy.Signature):
                """从文本中提取结构化信息"""
                text: str = dspy.InputField(desc="输入文本")
                output: str = dspy.OutputField(desc="JSON格式的结构化输出")
            
            # 创建模块
            extractor = dspy.ChainOfThought(ExtractSignature)
            
            # 执行抽取
            result = extractor(text=text)
            
            # 解析 JSON
            try:
                return json.loads(result.output)
            except:
                return {"raw_output": result.output}
        
        except ImportError:
            return {"error": "DSPy not installed. Run: pip install dspy-ai"}
        except Exception as e:
            return {"error": f"DSPy error: {e}"}
    
    def extract_with_guidance(
        self,
        text: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 Guidance 进行结构化抽取
        
        Args:
            text: 输入文本
            schema: 输出 schema 定义
            
        Returns:
            结构化数据
        """
        try:
            import guidance
            
            # 配置 LLM
            # llm = guidance.llms.OpenAI("gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
            
            # 定义 Guidance 程序
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            
            program = guidance("""
{{#system~}}
你是一个信息抽取专家。从给定的文本中提取结构化信息。
{{~/system}}

{{#user~}}
请从以下文本中提取信息，并按照以下 JSON schema 格式输出：

Schema:
{{schema}}

文本：
{{text}}

请只输出 JSON，不要包含其他解释。
{{~/user}}

{{#assistant~}}
{{gen 'output' temperature=0}}
{{~/assistant}}
""")
            
            # 执行
            # result = program(text=text, schema=schema_str, llm=llm)
            # return json.loads(result["output"])
            
            return {"error": "Guidance implementation needs LLM configuration"}
        
        except ImportError:
            return {"error": "Guidance not installed. Run: pip install guidance"}
        except Exception as e:
            return {"error": f"Guidance error: {e}"}
    
    def simple_extract(
        self,
        text: str,
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        简单的结构化抽取（使用 LLM API）
        
        Args:
            text: 输入文本
            fields: 要提取的字段列表
            
        Returns:
            结构化数据
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            fields_str = ", ".join(fields)
            prompt = f"""从以下文本中提取以下字段的信息：{fields_str}

文本：
{text}

请以 JSON 格式输出，格式如下：
{{
  "field1": "value1",
  "field2": "value2",
  ...
}}

只输出 JSON，不要包含其他内容："""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            return {"error": f"Extraction error: {e}"}


if __name__ == "__main__":
    # Day 11-13 示例：结构化输出
    print("=" * 50)
    print("Day 11-13: 结构化输出示例")
    print("=" * 50)
    
    demo = StructuredOutputDemo()
    
    # 测试文本
    text = """
    张三，男，35岁，是一名软件工程师。他于2020年加入阿里巴巴公司，负责开发AI系统。
    他的邮箱是zhangsan@example.com，电话是13800138000。
    他住在北京市海淀区中关村大街1号。
    """
    
    # 定义要提取的字段
    fields = ["姓名", "性别", "年龄", "职业", "公司", "邮箱", "电话", "地址"]
    
    print(f"\n输入文本:\n{text}")
    print(f"\n要提取的字段: {fields}")
    
    # 使用简单方法提取
    print("\n【结构化抽取结果】")
    result = demo.simple_extract(text, fields)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

