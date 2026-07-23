import os, asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from tools.index import internet_search, fetch_text_from_url
from prompt.index import SKILL_AGENT_SYSTEM_PROMPT
from mcp_tools.index import get_mcp_tools

chat_model = init_chat_model(os.environ["OPENAI_MODEL"])


async def get_agent():
    tools = [internet_search, fetch_text_from_url] + await get_mcp_tools(
        "C:/Users/zengd/Desktop"
    )
    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=SKILL_AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


async def main():
    agent = await get_agent()
    result = await agent.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "使用js语言在目录C:/Users/zengd/Desktop/Snake下生成一个贪吃蛇游戏",
                }
            ]
        },
        config={"configurable": {"thread_id": "1"}},
    )
    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    asyncio.run(main())
