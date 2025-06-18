#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import cmudict 
import nltk
import spacy
import os
import pandas as pd
import re
import token
import math
from pathlib import Path
from collections import Counter




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
    fk = 0.39*(words/sentences)+11.8*(syllables/words)-15.59
    return fk


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
    """parse: The goal of this function is to process the texts with spaCy’s tokenizer 9
    and parser, and store the processed texts. Your completed function should:
    i. Use the spaCy nlp method to add a new column to the dataframe that
    contains parsed and tokenized Doc objects for each text."""
    df["parsed"] = df["text"].apply(nlp)
    
    """ii. Serialise the resulting dataframe (i.e., write it out to disk) using the pickle
    format."""
    df.to_pickle(store_path/out_name)
    """iii. Return the dataframe."""
    return(df)
    """iv. Load the dataframe from the pickle file and use it for the remainder of this
    coursework part. Note: one or more of the texts may exceed the default
    """


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
    co_occurrence = Counter()
    subject_total = Counter()
    verb_count = 0
    total_tokens = len(doc)

    for word in doc:
        if word.dep_ in ["nsubj", "nsubjpass"]:
            subject_total[word.lemma_.lower()] += 1
        
        if word.lemma_ == target_verb and word.pos_ == "VERB":
            verb_count += 1
            for child in word.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    co_occurrence[child.lemma_.lower()] += 1

    pmi_scores = {}

    for subject in co_occurrence:
        p_subject = subject_total[subject] / total_tokens
        p_verb = verb_count / total_tokens
        p_joint = co_occurrence[subject] / total_tokens

        if p_subject > 0 and p_verb > 0 and p_joint > 0:
            pmi = math.log(p_joint / (p_subject * p_verb), 2)
            pmi_scores[subject] = round(pmi, 4)

    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = dict()
    for word in doc:
        if word.pos_ == "VERB" and word.lemma_ == verb:
            if word in token.lefts:
                if word.pos_ in ["NOUN", "PROPN", "PRON"] and word.dep_ in ["nsubj", "nsubjpass"]:
                    subj = word.lemma_.lower()
                    if subj not in subjects:
                        subjects[subj] = 1
                    else:
                        subjects[subj] += 1

    return dict(sorted(subjects.items(), key=lambda x: x[1], reverse=True)[:10])





def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = {}
    for word in doc:
        if word.pos_ == "ADJ":
            adj = word.lemma_.lower()
            if adj not in adjectives:
                adjectives[adj] = 1
            else:
                adjectives[adj] += 1

    return sorted(adjectives.items(), key=lambda x: x[1], reverse=True)[:10]


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
    #print(adjective_counts(df))
    
    """for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")"""
    

