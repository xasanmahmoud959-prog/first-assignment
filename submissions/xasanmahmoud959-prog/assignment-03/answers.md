

   Part A Research

1. Text Vectorization and One-Hot Encoding

Text vectorization converts text into numerical feature vectors that machine-learning algorithms can process. Documents must use the same feature dimensions and feature order so that a model interprets each position consistently.

One-hot encoding assigns a word a vector with one "1" and zeros elsewhere, based on a vocabulary.

For example .    using the vocabulary "[cash, class, free, lunch, notes, win]", the word "free" is represented as:

"[0, 0, 1, 0, 0, 0]"

One-hot encoding represents individual tokens. A multiword document requires multiple token vectors or an aggregation approach.

For example   a message classification system must represent a whole SMS, not just one isolated word.

2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) counts occurrences of vocabulary terms in documents.

- Rows represent documents.
- Columns represent features or terms.
- Cells represent word counts.

It is called a "bag" because unigram BoW ignores the original word order.

This is useful for tasks such as spam classification, where word occurrence can be informative. However, it can harm meaning-sensitive tasks.

For example, "I like it" and "I do not like it" differ in meaning, yet unigram BoW loses word order and may not model negation.

3. Vocabulary and Sparsity

Vocabulary size is the number of distinct features retained

Text matrices are often sparse because each document uses only a small fraction of the corpus-wide vocabulary

Sparse formats store nonzero entries and their locations rather than all zeros, saving memory

The following settings can change the vocabulary:

- "min_df" removes terms occurring in fewer than a specified number or fraction of documents
- "max_df" removes overly common terms.
- "max_features" caps the number of selected features.
- Lowercasing merges uppercase and lowercase variants.
- Stopword removal discards selected common words.
- Stemming reduces word forms toward their stems.

Each operation changes the vocabulary and can also discard useful information.

4. N-grams

An n-gram is a consecutive sequence of "n" tokens.

- Unigrams represent single words.
- Bigrams represent two-word sequences.
- Trigrams represent three-word sequences.

For example.   in the phrase "was not good", the bigram "not good" captures part of the phrase's polarity and word order.

Larger n-gram ranges create many more features, increasing memory and computation costs. They can also add rare patterns that may lead to overfitting.

5. TF-IDF

TF-IDF (term frequency–inverse document frequency) weights a term according to its frequency in a document and its rarity across the collection.

A simple textbook term frequency (TF) is the raw count:

"TF(t,d) = f(t,d)"

Textbook inverse document frequency (IDF) is:

"IDF(t) = ln(N / df(t))"

Where:

- "N" is the total number of documents.
- "df(t)" is the number of documents containing the term.

The TF-IDF formula is:

  "TF-IDF(t,d) = TF(t,d) × IDF(t)"

Rare terms receive higher IDF because they can help distinguish documents.

The inverse-document-frequency idea developed in information retrieval, including Karen Spärck Jones’ statistical work on term relevance.

Scikit-learn uses smoothed IDF by default:

"IDF(t) = ln((1 + N) / (1 + df(t))) + 1"

It then applies L2 normalization to each document vector.

Therefore, TF-IDF values are weights, not probabilities. They are not a probability distribution and do not need to sum to one.

6. Fit, Transform, and Data Leakage

- "fit" learns a vocabulary and any required corpus statistics.
- "transform" applies the learned representation to new text.
- "fit_transform" performs both operations on the provided data.

Out-of-vocabulary tokens are ignored by default.

Data leakage occurs when information from test data influences learned preprocessing. This can make evaluation less independent and potentially produce overly optimistic results.

The correct process is:

1. Split the data into training and test sets.
2. Fit the vectorizer on training text only.
3. Transform both training and test sets using the same fitted vectorizer.
4. Train and evaluate the model.

7. Case Study: Somali News Classification

Ali (2026) studied Somali news topic classification using 7,240 manually curated articles from BBC Somali and VOA Somali, divided into eight topics.

The pipeline used word-level unigram TF-IDF with a Linear SVM.

The study reported:

- 94.20% accuracy using a stratified 80/20 holdout.
- 93.11% mean accuracy using repeated holdout.
- 93.20% mean accuracy using five-fold cross-validation.

The results suggest that TF-IDF and a classical classifier can be an effective and resource-efficient baseline for Somali news classification.

However, unigram TF-IDF associates terms with classes rather than intrinsically understanding meaning. Synonyms are treated as distinct features, and word order, context, negation, and sarcasm are not adequately represented by unigram counts alone.

---

Part B – Practical

Task 2: Document-Term Matrix (D1–D5)

The following table shows the document-term matrix for documents D1–D5.

Document| are| before| but| call| cash| claim| class| entry| for| free| good| great| lecture| lunch| me| meeting| not| notes| now| phone| please| prizes| reply| send| still| the| to| today| was| we| were| win
D1| 0| 0| 0| 0| 0| 1| 0| 0| 0| 1| 0| 0| 0| 0| 0| 0| 0| 0| 1| 1| 0| 0| 1| 0| 0| 0| 1| 0| 0| 0| 0| 2
D2| 0| 0| 0| 1| 1| 0| 0| 1| 0| 1| 0| 0| 0| 0| 0| 0| 0| 0| 1| 0| 0| 1| 0| 0| 0| 0| 0| 1| 0| 0| 0| 1
D3| 1| 0| 0| 0| 0| 0| 0| 0| 1| 0| 0| 0| 0| 1| 0| 1| 0| 0| 0| 0| 0| 0| 0| 0| 1| 0| 0| 1| 0| 1| 0| 0
D4| 0| 1| 0| 0| 0| 0| 1| 0| 0| 0| 0| 0| 1| 0| 1| 0| 0| 1| 0| 0| 1| 0| 0| 1| 0| 0| 0| 0| 0| 0| 0| 0
D5| 0| 0| 1| 0| 0| 0| 0| 0| 0| 0| 1| 1| 1| 0| 0| 0| 1| 1| 0| 0| 0| 0| 0| 0| 0| 2| 0| 0| 1| 0| 1| 0

Task 7: D2 Counts and TF-IDF

The table below shows the word counts and TF-IDF weights for document D2.

Word| Count| TF-IDF
call| 1| 0.38914146
cash| 1| 0.38914146
entry| 1| 0.38914146
free| 1| 0.31395688
now| 1| 0.31395688
prizes| 1| 0.38914146
today| 1| 0.31395688
win| 1| 0.31395688

Task 8: Hand Calculations for "cash" and "today"

Document: D2
Total number of documents (N): 5

CountVectorizer tokenizes D2 as:

"free, entry, win, cash, prizes, today, call, now"

Therefore, D2 contains 8 tokens.

Both "cash" and "today" occur once in D2.

From the count matrix:

- "df(cash) = 1"
- "df(today) = 2"

Calculation Table

Term| Count in D2| Raw-count TF| Normalized TF (count/8)| DF| Textbook IDF ln(5/df)| TF-IDF (raw-count TF)| TF-IDF (normalized TF)| Scikit-learn D2 weight
cash| 1| 1| 0.125000| 1| 1.609438| 1.609438| 0.201180| 0.389141
today| 1| 1| 0.125000| 2| 0.916291| 0.916291| 0.114536| 0.313957

Hand Calculation

Using the normalized-TF convention in the script:

For cash:

"TF-IDF = (1/8) × ln(5/1)"

"TF-IDF = 0.125 × 1.609438"

"TF-IDF = 0.201180"

For today:

"TF-IDF = (1/8) × ln(5/2)"

"TF-IDF = 0.125 × 0.916291"

"TF-IDF = 0.114536"

If TF is defined as the raw count, the corresponding textbook values are 1.609438 and 0.916291.

Both conventions rank "cash" above "today", matching scikit-learn:

"0.389141 > 0.313957"

Scikit-learn's values differ because its default IDF is smoothed and each document vector is L2-normalized.

---

Short Task Answers

Task 1

Each word maps to a separate six-element vector because the vocabulary contains six words.

The sentence "win free cash" yields three vectors, one per token, rather than a single sentence vector.

One-hot encoding identifies vocabulary items but does not by itself aggregate them into a document representation.

Task 2

The matrix has one row per document and one column per vocabulary word, with cell counts.

In D1, "win" has a count of 2 because it appears in both "Win" and "WIN". Default lowercasing merges those case forms.

"win free phone" and "phone free win" produce identical vectors because Bag of Words ignores word order.

Task 4

Default lowercasing combines "Win" and "WIN", and punctuation is not retained as a feature. The default token pattern also excludes the single-character word "a".

With "lowercase=False", capitalized forms become separate tokens, and the uppercase "FREE" is present separately.

English stopwords remove words such as "a", "now", and "to". In D1–D5, "not" is also removed by the English stopword list, which is risky because D5 says "not good".

Task 5

"min_df=2" retained words occurring in at least two documents:

"free, lecture, notes, now, the, today, win"

"max_features=5" retained the five features selected by frequency. Ties may be resolved by feature ordering.

These settings remove rare terms or cap the feature space, but they can also discard useful distinctive terms.

Task 6

D5 bigrams include "not good", preserving the negation phrase.

The corpus vocabulary grew from 32 unigrams to 67 features with unigrams and bigrams, and 98 features with unigrams, bigrams, and trigrams.

The increase is substantial because every distinct contiguous phrase can become a feature.

Task 7

TF-IDF downweights terms common across documents and gives more relative importance to distinctive words.

In D2, "cash", "entry", "call", and "prizes" receive higher weights than "free", "now", "today", and "win" in this corpus because their document frequencies differ.

In D1, "win" has a count of two and gains a stronger normalized TF-IDF weight than its other terms.

Task 8

The hand calculation uses raw-count TF and textbook IDF "ln(N/df)", without smoothing or vector normalization.

"Cash" has "df=1" and "today" has "df=2", so "cash" has higher textbook IDF and TF-IDF.

This matches their underlying rarity ordering, although scikit-learn values use smoothing and L2 normalization.

Task 9: Sparse Matrix

The count matrix contains 5 documents and 32 vocabulary features.

Total number of cells:

"5 × 32 = 160"

Non-zero cells:

"37"

Zero cells:

"160 − 37 = 123"

Percentage of zeros:

"(123 / 160) × 100 = 76.875%"

Therefore, the matrix contains 160 cells, 37 non-zero cells, and 123 zero cells. The percentage of zero cells is 76.875%.

Task 10

The vectorizer fitted on D1–D4 does not know "prize", "your", "is", "moved", or "tomorrow". These words are ignored when transforming the new texts.

"prize" is distinct from "prizes" because no stemming or lemmatization is applied.

A stemming or lemmatization step could map related forms to a shared root or lemma if it is used consistently before fitting and transforming.

Task 11

Fitting on D1–D4 only keeps the test-only terms out of the vocabulary.

Fitting on the training data plus N1 and N2 introduces "is", "moved", "prize", "tomorrow", and "your". These are test-set vocabulary leakage for this experiment.

The vectorizer should be fitted on training text only.

Task 12

The vectors share "the", "food", and "was", so their pairwise cosine similarities are 0.75.

The similarity matrix does not rank R1 ("tasty") closer to R2 ("delicious") than R3 ("terrible"). It gives both 0.75 because those opinion words are different features and have no semantic relationships encoded.

Bag of Words can capture overlap, but not synonymy or sentiment meaning.

---

Spam Detector Settings

For a spam detector, I would start with word unigrams and bigrams using TF-IDF or counts, evaluated on a train/test split. I would compare both representations using the same classifier and metrics.

Bigrams may preserve useful phrases such as "claim prize" and "free entry", while TF-IDF may reduce the influence of common words.

I would carefully evaluate stopword removal rather than apply it blindly because terms such as "not" may affect intent.

I would use vocabulary limits only after checking the resulting performance and avoid fitting the vectorizer on test text.

Stemming could help merge variants like "prize" and "prizes", but its effect should be tested rather than assumed to improve accuracy.

---

References

Ali, M. M. (2026). Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus. Scientific Reports.
https://www.nature.com/articles/s41598-026-68159-5

Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. Journal of Documentation, 28(1), 11–21.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
https://nlp.stanford.edu/IR-book/

Scikit-learn developers. Feature extraction: text.
https://scikit-learn.org/stable/modules/feature_extraction.html

Scikit-learn developers. TfidfVectorizer.
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
