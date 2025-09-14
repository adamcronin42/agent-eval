"""User feedback tool for getting clarification from users."""

import os
from agent_eval.tools import Tool


class UserFeedbackTool(Tool):
    """Tool for asking users for clarification or additional information."""

    def get_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        return {
            "name": "ask_user",
            "description": "Ask the user for clarification or additional information when the request is unclear",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A clear, specific question to ask the user"
                    }
                },
                "required": ["question"]
            }
        }

    def execute(self, question: str) -> str:
        """
        Ask the user a question and return their response.

        In interactive mode: prompts user for input
        In evaluation mode: returns unavailable message
        """
        # Check if we're in evaluation mode
        if os.getenv("AUTO_APPROVE_TOOLS", "false").lower() == "true":
            return "User feedback not available in evaluation mode"

        # Interactive mode
        print(f"\n🤔 Agent Question: {question}")
        try:
            response = input("Your answer: ").strip()
            return response if response else "No response provided"
        except (EOFError, KeyboardInterrupt):
            return "User interaction cancelled"