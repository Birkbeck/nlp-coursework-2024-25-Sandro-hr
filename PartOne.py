#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import cmudict 
import nltk
import spacy
from pathlib import Path
import os
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000
d = cmudict.dict()


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    if word in d:
        word = word.lower()
        syllable = []
        pronounce = (d[word][0])
        for cluster in pronounce:
            if not cluster[-1].isalpha():
                syllable.append(cluster)
        syllables = len(syllable)
        return(syllables)

    else:
        cluster = re.findall(r'[aeiouy]+', word)
        return len(cluster)
        

#print(count_syl("elephant", d))

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = len(nltk.sent_tokenize(text))
    items = nltk.word_tokenize(text)
    words = []
    for item in items:
        if item.isalpha():
            words.append(item)
    syllables = 0
    for word in words:
        syllables += count_syl(word,d)
    words = len(words)
    #print(sentences,words,syllables)
    fk = 0.39*(words/sentences)+11.8*(syllables/words)-15.59
    return fk

print(fk_level("Hello my name is Sandro. I am twenty five years old and I am blonde. Thank you for coming to my ted talk.", d))





def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    
    data = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".txt"):
                path = os.path.join(root,name)
                title, author, year = name[:-4].split('-')
                title = title.replace('_', ' ')
                author = author.replace('_', ' ')
                year = int(year)
                with open(path, encoding='utf-8') as f:
                    text = f.read()
                data.append({
                    'text': text,
                    'title': title,
                    'author': author,
                    'year': year
                })
    df = pd.DataFrame(data)
    df = df.sort_values('year', ascending=True).reset_index(drop=True)
    return df



def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    tokens = []
    clean_tokens = []
    tokens.extend(nltk.word_tokenize(text))
    for token in tokens:
        if token.isalpha():
            clean_tokens.append(token.lower())
    
    ttr = len(set(clean_tokens)) / len(clean_tokens)
    return ttr



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

