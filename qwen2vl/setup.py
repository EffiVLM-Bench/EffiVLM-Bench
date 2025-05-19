from setuptools import setup, find_packages

setup(
    name="qwen2vl",  
    version="0.1.0",  
    description="A Python library for qwen2vl functionalities, incorporating modifications to qwen2vl with methods for KV cache compression. ", 
    packages=find_packages(),  
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',  
)
