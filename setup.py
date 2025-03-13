from setuptools import setup, find_packages

setup(
    name='Rufus',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langgraph",
        "langsmith",
        "langchain_groq",
        "langchain_community",
        "python-dotenv"
    ],
    author='Achala Pandit',  
    author_email='achala.s.pandit@gmail.com',
    description='A library for agentic web crawling.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='==3.9.5',

)