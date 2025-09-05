from setuptools import setup, find_packages

setup(
    name="agent-eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "tiktoken>=0.5.0",
        "requests>=2.31.0",
        "colorama>=0.4.6",
        "tabulate>=0.9.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "agent-eval=agent_eval.cli:cli",
        ],
    },
    python_requires=">=3.8",
)