from setuptools import setup, find_packages

setup(
    name="code-agent",
    version="0.1.0",
    description="A ReAct-based code agent with OpenAI integration",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.0.0",
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "code-agent=cli.main:cli",
        ],
    },
    python_requires=">=3.8",
)