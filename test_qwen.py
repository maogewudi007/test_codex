from langchain_openai import ChatOpenAI
import os
from langchain_openai import OpenAI


from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

# chatLLM = ChatTongyi(
#     api_key='sk-bb5511a45a1a4613a531ff77e37e9ffc',
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     # other params...
#     model_kwargs={
#         "enable_thinking":True
#     }
# )

chatLLM = ChatOpenAI(
    api_key='sk-no-key-required',
    base_url="http://172.16.140.213:9080/qwen32/v1",
    model="gpt-3.5-turbo",
    max_tokens=4000,
    streaming=True, 
    model_kwargs = {
        "enable_thinking": True,
        "return_thinking": True,
    }
)

completion = chatLLM.astream(
    [HumanMessage(content="你是谁")])
is_answering = False
print("="*20+"思考过程 "+"="*20)
for chunk in completion:
    if chunk.additional_kwargs.get("reasoning_content"):
        print(chunk.additional_kwargs.get("reasoning_content"),end="",flush=True)
    else:   
        if not is_answering:
            print("\n"+"="*20+"回复内容"+"="*20)
            is_answering = True
        print(chunk.content,end="",flush=True)

