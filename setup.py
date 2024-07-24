from setuptools import setup, find_packages

setup(
    name="civiverse_prompt_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'nltk',
        'torch',
        'sentence-transformers',
        'umap-learn',
        'bertopic',
        'numpy',
        'plotly',
        'spacy'
    ],
    entry_points={
        'console_scripts': [
            'run_analysis=src.analysis:main',
            'run_ner=src.ner_processing:main'
        ],
    },
)

