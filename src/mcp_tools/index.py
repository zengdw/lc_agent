import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_mcp_tools(root_path: str):
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    root_path,
                ],
                "encoding": "UTF-8",
            }
        }
    )

    tools = await client.get_tools()

    return tools


if __name__ == "__main__":
    tools = asyncio.run(get_mcp_tools("C:/Users/zengd/Desktop"))
    print(tools)
