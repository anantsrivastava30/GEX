from setuptools import setup, find_packages

setup(
    name="quant_analysis",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "requests",
        "openai",
        "tiktoken",
        "streamlit",
        "pandas",
        "pyarrow",
        "numpy",
        "plotly",
        "matplotlib",
        "seaborn",
        "yfinance",
        "feedparser",
        "supabase",
        "PyYAML",
        "tabulate",
    ],
    entry_points={
        "console_scripts": [
            "quant=quant_analysis.services.ai_analysis:main",
        ],
    },
)
