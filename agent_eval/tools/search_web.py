"""Web search tool using DuckDuckGo API."""

import requests
import json
from agent_eval.tools import Tool, ToolExecutionError


class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo's instant answer API."""

    def get_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        return {
            "name": "search_web",
            "description": "Search the web for current information, news, or facts not in training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - be specific and concise"
                    }
                },
                "required": ["query"]
            }
        }

    def execute(self, query: str) -> str:
        """
        Search the web using DuckDuckGo's instant answer API.

        Args:
            query: The search query string

        Returns:
            Formatted search results as a string
        """
        try:
            # Use DuckDuckGo Instant Answer API (no key required)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Format results
            result_parts = []
            result_parts.append(f"Search results for '{query}':\n")

            # Direct answer
            if data.get("Answer"):
                result_parts.append(f"Answer: {data['Answer']}")

            # Abstract summary
            if data.get("Abstract"):
                result_parts.append(f"Summary: {data['Abstract']}")

            # Abstract text (more detailed)
            if data.get("AbstractText") and data["AbstractText"] != data.get("Abstract", ""):
                result_parts.append(f"Details: {data['AbstractText']}")

            # Related topics (first 3)
            related_topics = data.get("RelatedTopics", [])
            if related_topics:
                result_parts.append("\nRelated information:")
                for i, topic in enumerate(related_topics[:3]):
                    if isinstance(topic, dict) and topic.get("Text"):
                        result_parts.append(f"{i+1}. {topic['Text']}")

            # If we have results, join them
            if len(result_parts) > 1:
                return "\n\n".join(result_parts)
            else:
                return f"No results found for: {query}"

        except requests.RequestException as e:
            raise ToolExecutionError(f"Search failed: {str(e)}")
        except json.JSONDecodeError:
            raise ToolExecutionError("Failed to parse search results")
        except Exception as e:
            raise ToolExecutionError(f"Search error: {str(e)}")