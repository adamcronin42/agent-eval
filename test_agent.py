#!/usr/bin/env python3
"""
Simple test script for the Core Agent.
"""

import os
from agent_eval.agent import Agent

def test_agent():
    """Test the agent with various prompts."""

    # Check if we have required environment variables
    print("🔍 Checking environment setup...")

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set. Please set it in .env file for gemini/gemini-2.5-flash")
        return False

    print("✅ Environment variables configured")

    try:
        # Initialize agent
        print("\n🤖 Initializing agent...")
        agent = Agent()
        print("✅ Agent initialized successfully")

        # Test 1: Simple question (no tools needed)
        print("\n📝 Test 1: Simple question (no tools)")
        result = agent.run("What is 2 + 2?")
        print(f"Response: {result['response']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Iterations: {result['metrics']['iterations']}")

        # Test 2: Current information (should use web search)
        print("\n📝 Test 2: Current information (should trigger web search)")
        agent.reset_conversation()
        result = agent.run("What's the weather like in New York City today?")
        print(f"Response: {result['response']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Iterations: {result['metrics']['iterations']}")

        # Test 3: Ambiguous question (should ask user)
        print("\n📝 Test 3: Ambiguous question (should trigger ask_user tool)")
        agent.reset_conversation()
        result = agent.run("Help me with my project")
        print(f"Response: {result['response']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Iterations: {result['metrics']['iterations']}")

        print("\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_agent()