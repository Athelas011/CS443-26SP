'''text_util.py
'''
import re
import tensorflow as tf

def tokenize_words(text):
    '''Transforms a string sentence into words. Replaces contractions with same word without the apostrophe.

    This method is pre-filled for you (shouldn't require modification).

    Parameters:
    -----------
    text: string. Sentence of text.

    Returns:
    -----------
    list of strings. Words in the sentence `text`.
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    # Replaces contractions with same word without the apostrophe.
    text = re.sub('([A-Za-z]+)[\'`’]([A-Za-z]+)', r'\1'r'\2', text)
    # Remove <br /> HTML tags
    text = re.sub(r'<br\s*/?>', '', text)
    # Now split up the words
    return pattern.findall(text.lower())


def clean_review(text):
    '''Removes HTML tags and non-printing characters from a raw review str.

    Only used with char-level model.

    (This method has been provided for you and should not require modification.)

    Parameters:
    -----------
    text: str.
        A single review represented as a single str.

    Returns:
    --------
    str.
        The single review str without HTML and non-printing chars.
    '''
    # Remove <br /> HTML tags
    text = re.sub(r'<br\s*/?>', '', text)
    # This regex keeps standard printable characters,
    # newlines, and tabs, but throws away the "invisible" junk.
    # [^\x20-\x7E\n\t] means "Anything NOT between Space and ~"
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    return text


def get_most_similar_words(k, word_str, all_embeddings, word_str2int, eps=1e-10):
    '''Get the `k` words to the word `word_str` that have the most similar embeddings in `all_embeddings`.
    Uses the cosine similarity metric.

    Parameters:
    -----------
    k: int.
        How many words with the most similar embeddings to find?
    word_str: str.
        The query word.
    all_embeddings: ndarray. shape=(M, H).
        The embeddings extracted from the trained CBOW network and converted to NumPy ndarray.
    word_str2int: Python dictionary.
        Maps word str -> int index in the vocab.
    eps: float.
        Small number to prevent potential division by 0 in the cosine similarity metric.

    Returns:
    --------
    ndarray. shape=(k+1,)
        The indices in the vocab of the query word itself + those of the k most similar words in the vocab.
    ndarray. shape=(k+1,)
        The cosine similarity of the query word to itself + that of the k most similar words in the vocab.

    NOTE: the top_k TensorFlow function should be very helpful here.
    https://www.tensorflow.org/api_docs/python/tf/math/top_k
    -
    '''
    pass


def find_unique_word_counts(corpus, sort_by_count=True):
    '''Determine the number of unique words in the corpus along with the word counts.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    corpus: Python list of lists of str.
        List of sentences, each of which is a list of words (str).
    sort_by_count: bool.
        Whether to sort the words according to their frequency.

    Returns:
    --------
    Python dictionary. str->int.
        Maps the unique words (key) to their associated count in the corpus (value).

    '''
    unique_word_counts = {}
    for sent in corpus:
        for word in sent:
            if word not in unique_word_counts:
                unique_word_counts[word] = 1
            else:
                unique_word_counts[word] += 1

    if sort_by_count:
        unique_word_counts = {key: val for key, val in sorted(unique_word_counts.items(), key=lambda item: item[1],
                                                              reverse=True)}

    return unique_word_counts


def make_train_val_split(corpus, prop_val=0.2):
    '''Subdivides the provided corpus into a (smaller) training set and a validation set, composed of the last
    `prop_val` proportion of samples in x_train/y_train.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    corpus: Python list of str.
        Each review represented by a single str.
    prop_val: float.
        Proportion of the original training set to reserve for the validation set.

    Returns:
    --------
    Python list of str.
        Reviews in the training set.
    Python list of str.
        Reviews in the validation set.
    '''
    N = len(corpus)
    N_val = int(prop_val*N)
    N_train = N - N_val

    train = corpus[:N_train]
    val = corpus[N_train:]

    return train, val

def decode_special_tokens(seq):
    '''Helper test code function to visualize the presence of the pad, start, and end non-printing chars.'''
    seq = seq.replace('\x00', '<PAD>')
    seq = seq.replace('\x02', '<START>')
    seq = seq.replace('\x03', '<END>')
    return seq
