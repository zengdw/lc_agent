import os, asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
from prompt.index import SKILL_AGENT_SYSTEM_PROMPT
from mcp_tools.index import get_mcp_tools
from skills.index import list_available_skills, load_skill, get_skill_file_path
from tools.index import run_shell_command, get_current_time

chat_model = init_chat_model(os.environ["OPENAI_MODEL"])


async def get_agent():
    tools = [
        list_available_skills,
        load_skill,
        run_shell_command,
        get_current_time,
        get_skill_file_path,
    ] + await get_mcp_tools("C:/Users/zengd/Desktop")
    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=SKILL_AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


async def main():
    agent = await get_agent()
    async for chunk, metadata in agent.astream(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "今天是几月几号",
                }
            ]
        },
        config={"configurable": {"thread_id": "1"}},
        stream_mode="messages",
    ):
        if isinstance(chunk, BaseMessage) and chunk.content:
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
