Part A: Research Questions

Q1. How does text preprocessing influence the quality of machine learning models?

Text preprocessing influences machine learning models by changing the way information is represented before it is used for training Poorly prepared data can introduce noise inconsistent representations, and unnecessary features

For example a dataset containing customer reviews may include misspelled words repeated characters, unusual symbols or inconsistent formats. These issues may affect how a machine learning model learns patterns

However    preprocessing does not always improve performance. Some information removed during preprocessing may be useful for a specific task. Therefore, preprocessing techniques should be selected according to the dataset, the task and the model being used

Q2. What is information loss in text preprocessing, and why can it be a problem?

Information loss occurs when preprocessing removes or changes information that may be meaningful to a machine learning task.

For example, in a medical text classification system the expressions "symptoms present" and "no symptoms present" have different meanings. If a preprocessing operation removes a word that expresses negation, the resulting text may become misleading.

Similarly, removing numbers from financial documents could eliminate important information such as prices, percentages, dates, or transaction amounts.

The goal of preprocessing should therefore be to improve data representation without unnecessarily destroying useful information.

Q3. What is the relationship between tokenization design and languages with complex word structures?

Tokenization design is important because languages differ in how words are formed, written, and separated. Some languages use spaces to separate words, while others may combine several meaningful components into a single written word.

Languages with complex morphological structures may require tokenizers that can represent prefixes, suffixes, roots, and other word components effectively. A tokenizer that works well for one language may not work equally well for another.

Subword tokenization methods, such as Byte-Pair Encoding (BPE), WordPiece, and Unigram, are widely used in modern NLP systems because they can represent words that are not present in a model's vocabulary as complete words.

However, tokenization quality should be evaluated using the target language and task rather than assuming that one tokenizer is suitable for every language.

Q4. Why should preprocessing strategies be designed according to the domain of the dataset?

Preprocessing should be adapted to the domain because the same text feature may have different importance in different fields.

For example, dates, article numbers, and section references may be important in legal documents. In financial documents, currencies, numerical values, and percentages may be essential. In medical documents, abbreviations and dosage information may affect interpretation. In social media analysis, emojis, repeated characters, and informal expressions may carry emotional information.

A general preprocessing pipeline may not be appropriate for every dataset. Domain-specific preprocessing helps preserve information relevant to the task and avoid unnecessary modifications.

Q5. What are the trade-offs between preprocessing techniques and machine learning performance?

Preprocessing techniques involve trade-offs between reducing unnecessary information and preserving meaningful features.

Preprocessing decision| Potential benefit| Potential drawback
Removing numerical information| Reduces irrelevant numerical features| May remove prices, dates or quantities
Normalizing unusual text formats| Creates more consistent input| May hide differences between original expressions
Removing domain-specific terms| May simplify the dataset| May remove important technical vocabulary
Combining preprocessing methods| May improve model performance| Interactions between methods may produce unexpected results
Keeping original text| Preserves original information| May retain noise or inconsistencies

Preprocessing methods should be compared experimentally. Researchers can evaluate different preprocessing pipelines using the same dataset, model architecture, training conditions, and evaluation metrics to determine which approach performs best.

Q6. How can regular expressions be used to identify specific patterns in text data?

Regular expressions (Regex) can be used to identify patterns in unstructured text, including identifiers, dates, email addresses, numbers, and other structured elements.

Pattern| Purpose| Example
"\b[A-Z]{2,5}-\d{3,6}\b"| Identifies reference IDs| AB-2024
"\b\d{4}-\d{2}-\d{2}\b"| Identifies dates in YYYY-MM-DD format| 2026-11-22
"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"| Identifies email addresses| student@example.com
"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"| Identifies numbers with commas or decimals| 12,500.75

Regex can be used to extract, identify, validate, or selectively replace text patterns. It is useful when a dataset contains predictable structures that can be described using rules.

Q7. Case Study: How preprocessing affects Transformer and traditional machine learning models

Research paper:

Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. Information Systems, 121, 102342.

Research objective:

The researchers investigated whether preprocessing still affects classification performance when using modern Transformer models and traditional machine learning models.

Methodology:

The researchers tested preprocessing techniques and combinations of these techniques across four publicly available datasets from different domains. They evaluated nine machine learning models, including Transformer-based architectures.

Findings:

Preprocessing can significantly influence the performance of modern Transformer models. Different datasets and models respond differently to preprocessing. In certain experiments, preprocessing choices produced accuracy differences of up to 25%. A traditional Naïve Bayes classifier with an appropriate preprocessing strategy outperformed a Transformer in one reported comparison. The best preprocessing strategy depends on the task, dataset, and model.

Conclusion:

The researchers concluded that preprocessing remains an important part of NLP. Modern models do not eliminate the need to carefully evaluate text preparation methods. Researchers should select preprocessing strategies according to experimental evidence instead of applying the same techniques to every dataset.

References

1. Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. Information Systems, 121, 102342. https://doi.org/10.1016/j.is.2023.102342

2. Goldman, O., et al. (2024). Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.134/

3. Schmidt, C. W., et al. (2024). Tokenization Is More Than Compression. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.40/

4. Hugging Face. Normalization and pre-tokenization. https://huggingface.co/docs/course/main/chapter6/4






import re
import nltk
from collections import Counter

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree


# ==========================================
# ASSIGNMENT 2 - PREPROCESSING
# ==========================================

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"

T2 = "I don't think this jacket is worth the price. Not happy at all 😞"

T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."

T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

words = [
    "connection", "connected", "connecting",
    "organization", "organized", "university",
    "universal", "flies", "wolves", "better", "ran"
]


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ==========================================
# TASK 1: LOWERcasing
# ==========================================

heading("TASK 1: LOWERcasing")

print("Before:", T1)

lower_text = T1.lower()

print("After:", lower_text)


# ==========================================
# TASK 2: WHITESPACE NORMALIZATION
# ==========================================

heading("TASK 2: WHITESPACE NORMALIZATION")

print("Before:", repr(T4))

normalized_T4 = " ".join(T4.split())

print("After:", repr(normalized_T4))


# ==========================================
# TASK 3: REGEX CLEANING
# ==========================================

heading("TASK 3: REGEX CLEANING")

pattern = r"[^a-z\s]"

regex_result = re.sub(pattern, "", lower_text)

deleted = re.findall(pattern, lower_text)

print("\nA. Removing characters")

print("Pattern:", pattern)
print("Before:", lower_text)
print("After:", regex_result)
print("Deleted characters:", deleted)
print("Unique deleted characters:", sorted(set(deleted)))
print("Deletion counts:", dict(Counter(deleted)))


print("\nB. Replacing important information")

replacement_text = lower_text

replacement_text = re.sub(
    r"\b\d{10}\b",
    "PHONE_NUMBER",
    replacement_text
)

replacement_text = re.sub(
    r"£\s?[\d,]+",
    "MONEY",
    replacement_text
)

replacement_text = re.sub(
    r"!{2,}",
    "!",
    replacement_text
)

print("Before:", lower_text)
print("After:", replacement_text)


# ==========================================
# TASK 4: TOKENIZATION
# ==========================================

heading("TASK 4: TOKENIZATION")

print("\nA. T2 using split()")

tokens_split = T2.split()

print("Tokens:", tokens_split)
print("Token count:", len(tokens_split))


print("\nB. T2 using word_tokenize()")

tokens_word = word_tokenize(T2)

print("Tokens:", tokens_word)
print("Token count:", len(tokens_word))


print("\nC. T4 tokenization")

tokens_T4 = word_tokenize(T4)

print("Tokens:", tokens_T4)
print("Token count:", len(tokens_T4))
print("Vocabulary size:", len(set(tokens_T4)))


# ==========================================
# TASK 5: STOPWORDS AND NEGATION
# ==========================================

heading("TASK 5: STOPWORDS AND NEGATION")

english_stopwords = set(stopwords.words("english"))

tokens_T2 = word_tokenize(T2)

print("\nOriginal tokens:")
print(tokens_T2)

default_filtered = [
    word for word in tokens_T2
    if word.lower() not in english_stopwords
]

print("\nAfter NLTK default stopwords:")
print(default_filtered)

negation_words = {"not", "n't", "no", "never"}

custom_stopwords = english_stopwords - negation_words

custom_filtered = [
    word for word in tokens_T2
    if word.lower() not in custom_stopwords
]

print("\nAfter custom stopwords:")
print(custom_filtered)


# ==========================================
# TASK 6: STEMMING
# ==========================================

heading("TASK 6: STEMMING")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

stem_results = {}

print(
    f"{'WORD':<15}"
    f"{'PORTER':<15}"
    f"{'LANCASTER':<15}"
    f"{'SNOWBALL':<15}"
)

print("-" * 60)

for word in words:
    p = porter.stem(word)
    l = lancaster.stem(word)
    s = snowball.stem(word)

    stem_results[word] = {
        "Porter": p,
        "Lancaster": l,
        "Snowball": s
    }

    print(f"{word:<15}{p:<15}{l:<15}{s:<15}")


print("\nDifferent words merged into the same stem:")

for stemmer_name in ["Porter", "Lancaster", "Snowball"]:

    groups = {}

    for word in words:
        stem = stem_results[word][stemmer_name]
        groups.setdefault(stem, []).append(word)

    for stem, original_words in groups.items():
        if len(original_words) > 1:
            print(stemmer_name, ":", original_words, "->", stem)


print("\nStems produced by each algorithm:")

for word in words:
    print(
        word,
        "-> Porter:", porter.stem(word),
        "| Lancaster:", lancaster.stem(word),
        "| Snowball:", snowball.stem(word)
    )


# ==========================================
# TASK 7: LEMMATIZATION
# ==========================================

heading("TASK 7: LEMMATIZATION")

lemmatizer = WordNetLemmatizer()

pos_hints = {
    "connection": "n",
    "connected": "v",
    "connecting": "v",
    "organization": "n",
    "organized": "v",
    "university": "n",
    "universal": "a",
    "flies": "n",
    "wolves": "n",
    "better": "a",
    "ran": "v"
}

lemma_results = {}

print(
    f"{'WORD':<15}"
    f"{'DEFAULT':<15}"
    f"{'WITH POS HINT':<20}"
)

print("-" * 50)

for word in words:

    default_lemma = lemmatizer.lemmatize(word)

    hinted_lemma = lemmatizer.lemmatize(
        word,
        pos=pos_hints[word]
    )

    lemma_results[word] = (default_lemma, hinted_lemma)

    print(
        f"{word:<15}"
        f"{default_lemma:<15}"
        f"{hinted_lemma:<20}"
    )


print("\nWords changed by POS hints:")

for word, (default, hinted) in lemma_results.items():

    if default != hinted:
        print(word, ":", default, "->", hinted)


# ==========================================
# TASK 8: SENTENCE SPLITTING
# ==========================================

heading("TASK 8: SENTENCE SPLITTING")

print("Original text:")
print(T3)

sentences = sent_tokenize(T3)

print("\nSentences:")

for number, sentence in enumerate(sentences, start=1):
    print(number, sentence)

print("\nNumber of sentences:", len(sentences))


# ==========================================
# TASK 9: POS TAGGING AND NER
# ==========================================

heading("TASK 9: POS TAGGING AND NER")

tokens_T3 = word_tokenize(T3)

tagged_words = pos_tag(tokens_T3)

print("\nPOS TAGS:")

for item in tagged_words:
    print(item)


print("\nNAMED ENTITIES:")

ner_tree = ne_chunk(tagged_words)

print(ner_tree)


def extract_entities(tree):

    entities = []

    for node in tree:

        if isinstance(node, Tree):

            name = " ".join(
                word for word, tag in node.leaves()
            )

            label = node.label()

            entities.append((name, label))

            entities.extend(extract_entities(node))

    return entities


entities = extract_entities(ner_tree)

print("\nENTITY TABLE DATA:")

if entities:
    for name, label in entities:
        print("Entity:", name, "| Label:", label)
else:
    print("No entities detected.")


# ==========================================
# TASK 10: FULL PREPROCESSING PIPELINE
# ==========================================

heading("TASK 10: FULL PREPROCESSING PIPELINE")


def run_pipeline(text, stopword_list, pipeline_name):

    print("\n" + "=" * 55)
    print(pipeline_name)
    print("=" * 55)

    # Stage 1: Lowercase
    stage1 = text.lower()

    print("\nStage 1 - Lowercase:")
    print(stage1)

    # Stage 2: Regex
    stage2 = re.sub(r"[^\w\s']", "", stage1)

    print("\nStage 2 - Regex:")
    print(stage2)

    # Stage 3: Tokenization
    stage3 = word_tokenize(stage2)

    print("\nStage 3 - Tokenization:")
    print(stage3)

    # Stage 4: Stopwords
    stage4 = [
        word for word in stage3
        if word.lower() not in stopword_list
    ]

    print("\nStage 4 - Stopwords:")
    print(stage4)

    # Stage 5: Lemmatization
    tagged = pos_tag(stage4)

    def get_wordnet_pos(tag):

        if tag.startswith("J"):
            return "a"

        if tag.startswith("V"):
            return "v"

        if tag.startswith("N"):
            return "n"

        if tag.startswith("R"):
            return "r"

        return "n"

    stage5 = [
        lemmatizer.lemmatize(
            word,
            pos=get_wordnet_pos(tag)
        )
        for word, tag in tagged
    ]

    print("\nStage 5 - Lemmatization:")
    print(stage5)

    return stage5


# Pipeline A: Default NLTK stopwords

default_pipeline = run_pipeline(
    T2,
    english_stopwords,
    "PIPELINE A: DEFAULT STOPWORDS"
)


# Pipeline B: Custom stopwords preserving negation

custom_pipeline = run_pipeline(
    T2,
    custom_stopwords,
    "PIPELINE B: NEGATION-KEEPING STOPWORDS"
)


print("\nFINAL PIPELINE COMPARISON")

print("Default:", default_pipeline)
print("Custom:", custom_pipeline)

print("\nEND OF ASSIGNMENT 2")


