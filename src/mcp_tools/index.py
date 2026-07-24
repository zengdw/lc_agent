import asyncio, os
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
            },
            "tavily-mcp": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "tavily-mcp@latest"],
                "env": {
                    "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
                    "DEFAULT_PARAMETERS": '{"include_images": true, "max_results": 15, "search_depth": "advanced"}',
                },
            },
        }
    )

    tools = await client.get_tools()

    return tools


if __name__ == "__main__":
    tools = asyncio.run(get_mcp_tools("C:/Users/zengd/Desktop"))
    print(tools)
