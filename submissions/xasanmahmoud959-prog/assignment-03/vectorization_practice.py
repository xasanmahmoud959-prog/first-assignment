
import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# ASSIGNMENT 3 – VECTORIZATION
# PART B – PRACTICAL
# ============================================================


# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

D1 = "Win a FREE phone now! Reply WIN to claim."
D2 = "Free entry: win cash prizes today. Call now."
D3 = "Are we still meeting for lunch today?"
D4 = "Please send me the lecture notes before class."
D5 = "The lecture was not good, but the notes were great."

documents = [D1, D2, D3, D4, D5]
names = ["D1", "D2", "D3", "D4", "D5"]

N1 = "Claim your free prize now"
N2 = "Is the lunch meeting moved to tomorrow?"

test_docs = [N1, N2]

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

review_names = ["R1", "R2", "R3"]


# ------------------------------------------------------------
# 2. HELPER FUNCTION FOR SECTION HEADINGS
# ------------------------------------------------------------

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# TASK 1 – ONE-HOT ENCODING
# ============================================================

section("TASK 1 – ONE-HOT ENCODING")

onehot_words = ["free", "win", "phone", "cash"]

print("Selected vocabulary:", onehot_words)

for word in onehot_words:

    vector = [
        int(
            word in doc.lower()
            .replace("!", "")
            .replace(".", "")
            .replace(":", "")
            .replace("?", "")
            .split()
        )
        for doc in documents
    ]

    print(f"{word}: {vector}")


# ============================================================
# TASK 2 – BAG OF WORDS / DOCUMENT-TERM MATRIX
# ============================================================

section("TASK 2 – COUNT VECTORIZATION")

count_vec = CountVectorizer()

X_count = count_vec.fit_transform(documents)

vocab = count_vec.get_feature_names_out()

print("Vocabulary (alphabetical):", vocab.tolist())
print("Matrix shape:", X_count.shape)

print("\nDocument-term matrix:")
print("Rows: D1-D5")
print("Columns: vocabulary order\n")

print("        " + " ".join(f"{word:>10}" for word in vocab))

for name, row in zip(names, X_count.toarray()):

    print(
        f"{name:<6} "
        + " ".join(f"{int(value):>10}" for value in row)
    )

win_index = count_vec.vocabulary_["win"]

print("\nD1 count for 'win':", int(X_count[0, win_index]))

vector_1 = count_vec.transform(
    ["win free phone"]
).toarray()[0]

vector_2 = count_vec.transform(
    ["phone free win"]
).toarray()[0]

print("Vector for 'win free phone':", vector_1.tolist())
print("Vector for 'phone free win':", vector_2.tolist())


# ============================================================
# TASK 3 – TOKEN COUNTS AND VOCABULARY
# ============================================================

section("TASK 3 – TOKEN COUNTS AND VOCABULARY")

analyzer = count_vec.build_analyzer()

total_tokens = 0

for name, doc in zip(names, documents):

    tokens = analyzer(doc)

    total_tokens += len(tokens)

    print(f"{name}:")
    print("  Token count:", len(tokens))
    print("  Tokens:", tokens)

print("\nTotal tokens across documents:", total_tokens)
print("Unique vocabulary size:", len(vocab))


# ============================================================
# TASK 4 – PREPROCESSING
# ============================================================

section("TASK 4 – PREPROCESSING")

sample = "FREE prizes, TODAY!"

default_analyzer = CountVectorizer().build_analyzer()

print("Original text:", sample)

print(
    "Tokens after default preprocessing:",
    default_analyzer(sample)
)

print("\nExplanation:")
print("Default preprocessing:")
print("- Converts text to lowercase.")
print("- Removes punctuation.")
print("- Tokenizes the words.")


# ============================================================
# TASK 5 – VOCABULARY LIMITS
# ============================================================

section("TASK 5 – VOCABULARY LIMITS")

# min_df = 2

min_df_vec = CountVectorizer(min_df=2)

X_min_df = min_df_vec.fit_transform(documents)

print("min_df=2 vocabulary:")
print(min_df_vec.get_feature_names_out().tolist())

print("min_df=2 matrix shape:", X_min_df.shape)


# max_features = 5

limited_vec = CountVectorizer(max_features=5)

X_limited = limited_vec.fit_transform(documents)

print("\nmax_features=5 vocabulary:")
print(limited_vec.get_feature_names_out().tolist())

print("max_features=5 matrix shape:", X_limited.shape)


# ============================================================
# TASK 6 – N-GRAMS
# ============================================================

section("TASK 6 – N-GRAMS")

ngram_ranges = [
    (1, 1),
    (1, 2),
    (1, 3)
]

for ngram_range in ngram_ranges:

    vec = CountVectorizer(
        ngram_range=ngram_range
    )

    matrix = vec.fit_transform(documents)

    features = vec.get_feature_names_out()

    print(f"\nngram_range={ngram_range}")

    print("Vocabulary size:", len(features))

    print("Vocabulary:", features.tolist())


# ============================================================
# TASK 7 – TF-IDF
# ============================================================

section("TASK 7 – TF-IDF")

tfidf_vec = TfidfVectorizer()

X_tfidf = tfidf_vec.fit_transform(documents)

tfidf_vocab = tfidf_vec.get_feature_names_out()

print("TF-IDF vocabulary:", tfidf_vocab.tolist())

print("\nTF-IDF matrix:")
print("Rows: D1-D5")
print("Columns: vocabulary order\n")

print(
    "        "
    + " ".join(f"{word:>10}" for word in tfidf_vocab)
)

for name, row in zip(names, X_tfidf.toarray()):

    print(
        f"{name:<6} "
        + " ".join(f"{value:>10.4f}" for value in row)
    )


# D2 counts and TF-IDF weights

d2_counts = count_vec.transform([D2]).toarray()[0]

d2_tfidf = X_tfidf.toarray()[1]

print("\nD2: Count and TF-IDF weight")

for word, count, weight in zip(
    vocab,
    d2_counts,
    d2_tfidf
):

    if count > 0:

        print(
            f"{word}: "
            f"count={int(count)}, "
            f"tfidf={weight:.8f}"
        )


# ============================================================
# TASK 8 – HAND CALCULATIONS
# ============================================================

section("TASK 8 – HAND CALCULATIONS FOR CASH AND TODAY")

analyzer = CountVectorizer().build_analyzer()

d2_tokens = analyzer(D2)

print("D2 tokens:", d2_tokens)
print("D2 total token count:", len(d2_tokens))


for word in ["cash", "today"]:

    # Count occurrences in D2
    tf_count = d2_tokens.count(word)

    # Find the word's column in the count matrix
    word_col = count_vec.vocabulary_[word]

    # Calculate document frequency
    df = int(
        (X_count[:, word_col] > 0).sum()
    )

    # Calculate term frequency
    raw_tf = tf_count

    normalized_tf = tf_count / len(d2_tokens)

    # Calculate textbook IDF
    idf = np.log(len(documents) / df)

    # Calculate textbook TF-IDF
    raw_tf_idf = raw_tf * idf

    normalized_tf_idf = normalized_tf * idf

    # Get scikit-learn TF-IDF value
    sklearn_weight = float(
        X_tfidf[1, tfidf_vec.vocabulary_[word]]
    )

    print(f"\nWord: {word}")

    print("D2 count:", tf_count)

    print("D2 total tokens:", len(d2_tokens))

    print("Raw-count TF:", raw_tf)

    print(
        f"Normalized TF: "
        f"{tf_count}/{len(d2_tokens)} "
        f"= {normalized_tf:.8f}"
    )

    print(f"Document frequency (DF): {df}")

    print(
        f"Textbook IDF: "
        f"ln({len(documents)}/{df}) "
        f"= {idf:.8f}"
    )

    print(
        "Textbook TF-IDF (raw-count TF):",
        f"{raw_tf_idf:.8f}"
    )

    print(
        "Textbook TF-IDF (normalized TF):",
        f"{normalized_tf_idf:.8f}"
    )

    print(
        "Scikit-learn TF-IDF:",
        f"{sklearn_weight:.8f}"
    )


print("\nRanking by textbook calculation: cash > today")
print("Ranking by scikit-learn: cash > today")
print("Do the rankings match? Yes.")


# ============================================================
# TASK 9 – SPARSE MATRIX STATISTICS
# ============================================================

section("TASK 9 – SPARSE MATRIX STATISTICS")

# Calculate total matrix cells

total_cells = (
    X_count.shape[0] * X_count.shape[1]
)

# Count non-zero cells

nonzero_cells = X_count.nnz

# Calculate zero cells

zero_cells = total_cells - nonzero_cells

# Calculate percentage of zeros

zero_percentage = (
    zero_cells / total_cells
) * 100

print("Matrix shape:", X_count.shape)

print("Total cells:", total_cells)

print("Non-zero cells:", nonzero_cells)

print("Zero cells:", zero_cells)

print(
    f"Percentage of zeros: {zero_percentage:.2f}%"
)


# ============================================================
# TASK 10 – FIT, TRANSFORM AND OUT-OF-VOCABULARY WORDS
# ============================================================

section("TASK 10 – FIT AND TRANSFORM / OOV")

train_docs = [D1, D2, D3, D4]

test_docs = [N1, N2]

train_vec = CountVectorizer()

# Fit on training data

X_train = train_vec.fit_transform(train_docs)

# Transform test data

X_test = train_vec.transform(test_docs)

print(
    "Training vocabulary:",
    train_vec.get_feature_names_out().tolist()
)


for name, text, row in zip(
    ["N1", "N2"],
    test_docs,
    X_test.toarray()
):

    print(f"\n{name} text: {text}")

    print(f"{name} vector: {row.tolist()}")

    tokens = train_vec.build_analyzer()(text)

    oov = [
        token
        for token in tokens
        if token not in train_vec.vocabulary_
    ]

    print(f"{name} tokens: {tokens}")

    print(f"{name} out-of-vocabulary words: {oov}")


print("\nWhy is 'prize' unknown when D2 contains 'prizes'?")

print(
    "CountVectorizer treats them as different exact tokens. "
    "It does not automatically reduce words to their root form."
)

print(
    "Stemming or lemmatization can map related words "
    "to a shared form."
)


# ============================================================
# TASK 11 – DATA LEAKAGE
# ============================================================

section("TASK 11 – DATA LEAKAGE")

# Fit using training documents only

train_only_vec = CountVectorizer()

train_only_vec.fit(train_docs)

# Fit using training and test documents together

combined_vec = CountVectorizer()

combined_vec.fit(train_docs + test_docs)


train_vocab = set(
    train_only_vec.get_feature_names_out()
)

combined_vocab = set(
    combined_vec.get_feature_names_out()
)

# Find words introduced by test documents

leaked_words = sorted(
    combined_vocab - train_vocab
)

print(
    "Vocabulary fitted on D1-D4 only:",
    sorted(train_vocab)
)

print(
    "\nVocabulary fitted on D1-D4 + N1 + N2:",
    sorted(combined_vocab)
)

print(
    "\nWords leaked from test set:",
    leaked_words
)

print("\nExplanation:")

print(
    "Fitting on test documents allows test-set information "
    "to influence the learned vocabulary before evaluation."
)


# ============================================================
# TASK 12 – COSINE SIMILARITY AND MEANING
# ============================================================

section("TASK 12 – COSINE SIMILARITY AND MEANING")

review_vec = CountVectorizer()

X_reviews = review_vec.fit_transform(reviews)

print(
    "Review vocabulary:",
    review_vec.get_feature_names_out().tolist()
)


# Display review vectors

print("\nReview vectors:")

for name, row in zip(
    review_names,
    X_reviews.toarray()
):

    print(f"{name}: {row.tolist()}")


# Calculate cosine similarity

similarities = cosine_similarity(X_reviews)


# Display similarity matrix

print("\nCosine similarity matrix:")

print("       R1       R2       R3")

for name, row in zip(
    review_names,
    similarities
):

    print(
        f"{name}  "
        + "  ".join(
            f"{value:.6f}"
            for value in row
        )
    )


print(
    "\nR1 vs R2:",
    f"{similarities[0, 1]:.6f}"
)

print(
    "R1 vs R3:",
    f"{similarities[0, 2]:.6f}"
)


print("\nAnswer:")

print(
    "No. R1 is not more similar to R2 than to R3 "
    "according to Bag of Words cosine similarity."
)

print(
    "Both pairs share 'the' and 'food'. "
    "The vectorizer does not know that 'tasty' and "
    "'delicious' have similar meanings or that "
    "'terrible' has an opposite meaning."
)


# ============================================================
# END OF ASSIGNMENT 3
# ============================================================

section("ASSIGNMENT 3 – COMPLETED")
print("All 12 practical tasks have been executed.")
