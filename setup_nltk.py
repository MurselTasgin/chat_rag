# /Users/murseltasgin/projects/chat_rag/setup_nltk.py
"""
Script to download required NLTK data
"""
import nltk

print("Downloading required NLTK data...")

resources = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords')
]

for path, name in resources:
    try:
        nltk.data.find(path)
        print(f"✓ {name} already installed")
    except LookupError:
        print(f"⬇ Downloading {name}...")
        nltk.download(name, quiet=False)
        print(f"✓ {name} downloaded")

print("\n✓ All NLTK resources ready!")

