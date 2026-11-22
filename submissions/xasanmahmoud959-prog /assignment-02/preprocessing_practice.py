
preprocessing_practice.py

import re
import nltk

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ==================================================
# 0. DOWNLOAD REQUIRED NLTK DATA
# ==================================================

print("=" * 65)
print("NLP PREPROCESSING PRACTICE - ASSIGNMENT 2")
print("=" * 65)

required_data = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words"
]

for item in required_data:
    nltk.download(item, quiet=True)

print("\nNLTK DATA READY!")


# ==================================================
# INPUT TEXTS AND WORDS
# ==================================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

word_list = [
    "connection",
    "connected",
    "connecting",
    "organization",
    "organized",
    "university",
    "universal",
    "flies",
    "wolves",
    "better",
    "ran"
]


# ==================================================
# TASK 1 - LOWERCASE
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 1: LOWERCASE")
print("=" * 65)

print("BEFORE:")
print(T1)

lower_t1 = T1.lower()

print("\nAFTER:")
print(lower_t1)


# ==================================================
# TASK 2 - WHITESPACE NORMALIZATION
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 2: WHITESPACE NORMALIZATION")
print("=" * 65)

print("BEFORE:")
print(repr(T4))

normalized_t4 = re.sub(r"\s+", " ", T4).strip()

print("\nAFTER:")
print(repr(normalized_t4))


# ==================================================
# TASK 3 - REGEX CLEANING
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 3: REGEX CLEANING")
print("=" * 65)

# Rule 1: Remove every character except lowercase letters
# and whitespace.

pattern = r"[^a-z\s]"

regex_result = re.sub(pattern, "", lower_t1)

print("\nPART A: DELETE NON-LETTERS")

print("BEFORE:")
print(lower_t1)

print("\nPATTERN:")
print(pattern)

print("\nAFTER:")
print(regex_result)

# Identify the types of characters removed.
deleted_characters = sorted(set(re.findall(pattern, lower_t1)))

print("\nDELETED CHARACTER TYPES:")
print(deleted_characters)


# PART B: Replace important information instead of deleting it.

print("\nPART B: REPLACE IMPORTANT INFORMATION")

custom_regex = lower_t1

# Replace phone number.
custom_regex = re.sub(
    r"\b0\d{9}\b",
    "PHONE_NUMBER",
    custom_regex
)

# Replace money amount.
custom_regex = re.sub(
    r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    "MONEY",
    custom_regex
)

# Replace multiple exclamation marks with one.
custom_regex = re.sub(
    r"!{2,}",
    "!",
    custom_regex
)

print("BEFORE:")
print(lower_t1)

print("\nAFTER:")
print(custom_regex)


# ==================================================
# TASK 4 - TOKENIZATION
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 4: TOKENIZATION")
print("=" * 65)

# Tokenize T2 using split().
split_tokens = T2.split()

print("\nT2 USING split():")

print(split_tokens)
print("Token count:", len(split_tokens))


# Tokenize T2 using NLTK word_tokenize().
nltk_tokens = word_tokenize(T2)

print("\nT2 USING word_tokenize():")

print(nltk_tokens)
print("Token count:", len(nltk_tokens))


# Tokenize T4 and calculate vocabulary size.
t4_tokens = word_tokenize(normalized_t4.lower())

vocabulary = set(t4_tokens)

print("\nT4 TOKENS:")
print(t4_tokens)

print("T4 Token count:", len(t4_tokens))
print("T4 Vocabulary size:", len(vocabulary))
print("Unique tokens:", sorted(vocabulary))


# ==================================================
# TASK 5 - STOPWORDS AND NEGATION
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 5: STOPWORDS AND NEGATION")
print("=" * 65)

tokens = word_tokenize(T2.lower())

english_stopwords = set(stopwords.words("english"))

# Remove standard English stopwords.
without_stopwords = [
    token for token in tokens
    if token not in english_stopwords
]

print("\nORIGINAL TOKENS:")
print(tokens)

print("\nAFTER DEFAULT STOPWORD REMOVAL:")
print(without_stopwords)


# Build a custom stopword list that preserves negation.
negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

# Remove custom stopwords.
with_negation = [
    token for token in tokens
    if token not in custom_stopwords
]

print("\nAFTER NEGATION-KEEPING STOPWORD REMOVAL:")
print(with_negation)


# ==================================================
# TASK 6 - STEMMING
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 6: STEMMING")
print("=" * 65)

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

print("\nSTEMMER COMPARISON TABLE")

print(
    f"{'WORD':<15}"
    f"{'PORTER':<15}"
    f"{'LANCASTER':<15}"
    f"{'SNOWBALL':<15}"
)

print("-" * 60)

stem_results = {}

for word in word_list:

    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)

    stem_results[word] = {
        "Porter": porter_result,
        "Lancaster": lancaster_result,
        "Snowball": snowball_result
    }

    print(
        f"{word:<15}"
        f"{porter_result:<15}"
        f"{lancaster_result:<15}"
        f"{snowball_result:<15}"
    )


# Find different words that produce the same stem.
print("\nWORDS MERGED INTO THE SAME STEM:")

for stemmer_name in ["Porter", "Lancaster", "Snowball"]:

    groups = {}

    for word in word_list:
        stem = stem_results[word][stemmer_name]
        groups.setdefault(stem, []).append(word)

    found = False

    for stem, grouped_words in groups.items():

        if len(grouped_words) > 1:

            print(
                stemmer_name,
                "=>",
                grouped_words,
                "=>",
                stem
            )

            found = True

    if not found:
        print(stemmer_name, "=> No matching pair found")


# Find stems that are not English words.
english_vocabulary = {
    word.lower() for word in words.words()
}

print("\nSTEMS THAT ARE NOT REAL ENGLISH WORDS:")

for stemmer_name in ["Porter", "Lancaster", "Snowball"]:

    found = False

    for word in word_list:

        stem = stem_results[word][stemmer_name]

        if stem not in english_vocabulary:

            print(
                stemmer_name,
                ":",
                word,
                "->",
                stem
            )

            found = True

    if not found:
        print(stemmer_name, "=> No example found")


# ==================================================
# TASK 7 - LEMMATIZATION
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 7: LEMMATIZATION")
print("=" * 65)

lemmatizer = WordNetLemmatizer()

# Default POS is noun (n).
default_lemmas = {}

# Appropriate POS hints for the given words.
pos_hints = {
    "connection": "n",
    "connected": "v",
    "connecting": "v",
    "organization": "n",
    "organized": "v",
    "university": "n",
    "universal": "a",
    "flies": "v",
    "wolves": "n",
    "better": "a",
    "ran": "v"
}

hinted_lemmas = {}

print("\nLEMMATIZATION COMPARISON")

print(
    f"{'WORD':<15}"
    f"{'DEFAULT':<15}"
    f"{'WITH POS HINT':<20}"
    f"{'POS':<10}"
)

print("-" * 60)

for word in word_list:

    default_result = lemmatizer.lemmatize(word)

    pos = pos_hints[word]

    hinted_result = lemmatizer.lemmatize(
        word,
        pos=pos
    )

    default_lemmas[word] = default_result
    hinted_lemmas[word] = hinted_result

    print(
        f"{word:<15}"
        f"{default_result:<15}"
        f"{hinted_result:<20}"
        f"{pos:<10}"
    )


print("\nWORDS THAT CHANGED ONLY WITH POS HINT:")

for word in word_list:

    if default_lemmas[word] != hinted_lemmas[word]:

        print(
            word,
            ":",
            default_lemmas[word],
            "->",
            hinted_lemmas[word]
        )


# ==================================================
# TASK 8 - SENTENCE SPLITTING
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 8: SENTENCE SPLITTING")
print("=" * 65)

sentences = sent_tokenize(T3)

print("\nORIGINAL TEXT:")
print(T3)

print("\nSPLIT SENTENCES:")

for index, sentence in enumerate(sentences, start=1):

    print(f"Sentence {index}: {sentence}")

print("\nTotal sentences:", len(sentences))


# ==================================================
# TASK 9 - POS TAGGING AND NAMED ENTITY RECOGNITION
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 9: POS TAGGING AND NER")
print("=" * 65)

# Part A: POS tagging.
t3_tokens = word_tokenize(T3)

tagged_words = pos_tag(t3_tokens)

print("\nPOS TAGGING:")

for word, tag in tagged_words:

    print(f"{word:<15} {tag}")


# Part B: Named Entity Recognition.
named_entities = ne_chunk(tagged_words)

print("\nNAMED ENTITY TREE:")
print(named_entities)

# Extract entity names and labels.
entity_list = []

for subtree in named_entities:

    if hasattr(subtree, "label"):

        entity_name = " ".join(
            word for word, tag in subtree.leaves()
        )

        entity_label = subtree.label()

        entity_list.append(
            (entity_name, entity_label)
        )

print("\nDETECTED ENTITIES:")

if entity_list:

    for name, label in entity_list:

        print(f"Entity: {name:<20} Label: {label}")

else:

    print("No named entities detected.")


# Check expected entities in this sentence.
expected_entities = {
    "Mr. Smith": "PERSON",
    "London": "PLACE",
    "Friday": "DAY",
    "Apple": "ORGANIZATION"
}

detected_text = " ".join(
    name for name, label in entity_list
).lower()

print("\nEXPECTED ENTITY CHECK:")

for name, expected_label in expected_entities.items():

    found = name.lower() in detected_text

    print(
        f"{name:<15}"
        f"Expected: {expected_label:<15}"
        f"Recognized: {found}"
    )


# ==================================================
# TASK 10 - FULL PREPROCESSING PIPELINE
# ==================================================

print("\n\n" + "=" * 65)
print("TASK 10: FULL PREPROCESSING PIPELINE")
print("=" * 65)


def run_pipeline(text, stopword_list, pipeline_name):

    print("\n")
    print("-" * 65)
    print(pipeline_name)
    print("-" * 65)

    # STEP 1: Lowercase.
    stage1 = text.lower()

    print("\nSTEP 1 - LOWERCASE:")
    print(stage1)

    # STEP 2: Regex cleaning.
    # Preserve letters, whitespace and apostrophes.
    stage2 = re.sub(r"[^a-z\s']", " ", stage1)

    # Normalize whitespace.
    stage2 = re.sub(r"\s+", " ", stage2).strip()

    print("\nSTEP 2 - REGEX CLEANING:")
    print(stage2)

    # STEP 3: Tokenization.
    stage3 = word_tokenize(stage2)

    print("\nSTEP 3 - TOKENIZATION:")
    print(stage3)

    # STEP 4: Stopword removal.
    stage4 = [
        token for token in stage3
        if token not in stopword_list
    ]

  

    print("\nSTEP 4 - STOPWORD REMOVAL:")
    print(stage4)

    # STEP 5: Lemmatization.
    stage5 = [
        lemmatizer.lemmatize(token)
        for token in stage4
    ]

    print("\nSTEP 5 - LEMMATIZATION:")
    print(stage5)

    return stage5






# Pipeline A: Default English stopwords.
pipeline_default = run_pipeline(
    T2,
    english_stopwords,
    "PIPELINE A: DEFAULT STOPWORDS"
)


# Pipeline B: Custom stopwords preserving negation.
pipeline_custom = run_pipeline(
    T2,
    custom_stopwords,
    "PIPELINE B: NEGATION-KEEPING STOPWORDS"
)


# Final comparison.
print("\n\n" + "=" * 65)
print("FINAL PIPELINE COMPARISON")
print("=" * 65)

print("\nDEFAULT STOPWORDS:")
print(pipeline_default)

print("\nNEGATION-KEEPING STOPWORDS:")
print(pipeline_custom)

print("\nPREPROCESSING PRACTICE COMPLETED!")
print("=" * 65)
