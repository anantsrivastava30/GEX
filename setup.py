from setuptools import setup, find_packages

setup(
    name="quant_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "openai",
        "tiktoken",
        "streamlit",
        "pandas",
        "supabase",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "quant=quant_analysis.services.ai_analysis:main",
        ],
    },
)
