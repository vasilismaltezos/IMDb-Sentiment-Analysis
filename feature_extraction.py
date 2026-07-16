"""
Module gia tin epilogi idioteton me Information Gain kai tin diastasiologia ton dedomenwn
"""
import numpy as np
from collections import Counter


def calculate_entropy(labels):
    """Ypologizoume tin entropeia enos synolou etiketwn"""
    if len(labels) == 0:
        return 0.0

    counts = Counter(labels)
    total = len(labels)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * np.log2(prob)

    return entropy


def calculate_information_gain_fast(word_doc_matrix, labels, word_idx):
    """Ypologizoume to Information Gain mias leksis (grigoro me numpy)"""
    # Entropiia olikoy synoloy
    total_entropy = calculate_entropy(labels)

    # Pairnoume to binary vector gia ti leksi (poies reviews tin exoun)
    word_presence = word_doc_matrix[:, word_idx]

    # Xorizoume ta labels se ayta pou exoun kai den exoun ti leksi
    has_word_mask = word_presence == 1
    with_word = labels[has_word_mask]
    without_word = labels[~has_word_mask]

    # Ypologizoume vasismenes entropies
    n_total = len(labels)
    n_with = len(with_word)
    n_without = len(without_word)

    if n_with == 0 or n_without == 0:
        return 0.0

    entropy_with = calculate_entropy(with_word)
    entropy_without = calculate_entropy(without_word)

    # Vasismeni entropiia
    weighted_entropy = (n_with / n_total) * entropy_with + (n_without / n_total) * entropy_without

    # Information Gain
    info_gain = total_entropy - weighted_entropy

    return info_gain


def calculate_all_information_gains_vectorized(word_doc_matrix, labels):
    """
    Ypologizoume Information Gain gia OLES tis lexeis mazi (vectorized - POLY pio grigoro!)
    """
    labels = np.array(labels)
    n_samples = len(labels)
    n_words = word_doc_matrix.shape[1]

    # Entropiia olikoy synoloy
    total_entropy = calculate_entropy(labels)

    # Ypologismos gia ola ta words parallila
    information_gains = np.zeros(n_words)

    for word_idx in range(n_words):
        word_presence = word_doc_matrix[:, word_idx]

        has_word_mask = word_presence == 1
        with_word = labels[has_word_mask]
        without_word = labels[~has_word_mask]

        n_with = len(with_word)
        n_without = len(without_word)

        if n_with == 0 or n_without == 0:
            information_gains[word_idx] = 0.0
            continue

        entropy_with = calculate_entropy(with_word)
        entropy_without = calculate_entropy(without_word)

        weighted_entropy = (n_with / n_samples) * entropy_with + (n_without / n_samples) * entropy_without
        information_gains[word_idx] = total_entropy - weighted_entropy

    return information_gains


def select_features_by_information_gain(tokenized_texts, labels, filtered_vocab, m):
    """Epilogoume tis m lexeis me to megalytero Information Gain"""
    print(f"\nYpologismos Information Gain gia {len(filtered_vocab)} lexeis...")

    # BHMA 1: Dimiourgisi prosorinoy binary matrix gia oles tis filtered lexeis
    print("  Dimiourgia binary matrix...")
    vocab_list = list(filtered_vocab)

    vocab_temp_dict = {word: idx for idx, word in enumerate(vocab_list)}

    n_samples = len(tokenized_texts)
    n_words = len(vocab_list)

    # Dimiourgisi matrix gia taxitita - xrisimopoioume list comprehension
    word_doc_matrix = np.zeros((n_samples, n_words), dtype=np.int8)

    for i, tokens in enumerate(tokenized_texts):
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{n_samples} texts")
        token_set = set(tokens)
        for token in token_set:
            if token in vocab_temp_dict:
                word_doc_matrix[i, vocab_temp_dict[token]] = 1

    # BHMA 2: Ypologismos Information Gain gia OLES tis lexeis mazi (vectorized)
    print("  Ypologismos Information Gain (vectorized - grigoro)...")
    labels_array = np.array(labels)

    information_gains = calculate_all_information_gains_vectorized(word_doc_matrix, labels_array)

    # Dimiourgoume dictionary me scores
    word_ig_scores = {word: information_gains[i] for i, word in enumerate(vocab_list)}

    # Epilogi top-m lexewn
    sorted_words = sorted(word_ig_scores.items(), key=lambda x: x[1], reverse=True)
    top_m_words = [word for word, score in sorted_words[:m]]

    print(f"\nEpilegisan {len(top_m_words)} lexeis me to megalytero Information Gain")
    print(f"Top 10 lexeis:")
    for i, (word, score) in enumerate(sorted_words[:10]):
        print(f"  {i+1}. '{word}': {score:.4f}")

    return top_m_words


def build_vocabulary_dict(vocabulary):
    """Dimioyrgoume leksiko pou kanei map lexeis se indices"""
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    return vocab_dict


def texts_to_binary_features(tokenized_texts, vocab_dict):
    """Metatrepoume keimena se binary feature vectors"""
    n_samples = len(tokenized_texts)
    n_features = len(vocab_dict)

    X = np.zeros((n_samples, n_features), dtype=np.int8)

    for i, tokens in enumerate(tokenized_texts):
        for token in tokens:
            if token in vocab_dict:
                X[i, vocab_dict[token]] = 1

    return X


def extract_features(train_texts_tokenized, train_labels,
                     dev_texts_tokenized, test_texts_tokenized,
                     n_most_common, k_most_rare, m, min_doc_freq=5):
    """
    Ypologizoume to vocabulary kai metatrepoume ta keimena se features

    Returns:
        - vocabulary: lista me tis epilegmenes lexeis
        - X_train, X_dev, X_test: binary feature matrices
    """
    from .preprocessing import get_word_document_frequency, filter_vocabulary


    print("\n" + "="*50)
    print("EPILOGI IDIOTETON (FEATURES)")
    print("="*50)

    # Ypologismos sixnotitas lexewn
    word_doc_freq = get_word_document_frequency(train_texts_tokenized)

    # Afairesi sixnwn kai spaniwn lexewn
    filtered_vocab = filter_vocabulary(word_doc_freq, n_most_common, k_most_rare, len(train_texts_tokenized), min_doc_freq)

    # Epilogi m lexewn me Information Gain
    vocabulary = select_features_by_information_gain(
        train_texts_tokenized,
        train_labels,
        filtered_vocab,
        m
    )

    # Dmiourgia vocabulary dictionary
    vocab_dict = build_vocabulary_dict(vocabulary)

    print(f"\nTeliko megethos vocabulary: {len(vocab_dict)}")

    # Metatropi keimenwn se binary features
    print("\nMetatropi keimenwn se binary feature vectors...")
    X_train = texts_to_binary_features(train_texts_tokenized, vocab_dict)
    X_dev = texts_to_binary_features(dev_texts_tokenized, vocab_dict)
    X_test = texts_to_binary_features(test_texts_tokenized, vocab_dict)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    from collections import Counter
    return vocabulary, X_train, X_dev, X_test



def get_word_document_frequency(tokenized_texts):
    """
    Υπολογίζει document frequency:
    πόσα documents περιέχουν κάθε λέξη
    """
    df = Counter()
    for doc in tokenized_texts:
        for word in set(doc):
            df[word] += 1
    return df

    
