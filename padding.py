import numpy as np

from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

def _pad_token_sequences(sequences, max_tokens=None,
                         padding='pre', truncating='pre', value=0.):
    return keras_pad_sequences(sequences, maxlen=max_tokens, padding=padding, truncating=truncating, value=value)


def _pad_sent_sequences(sequences, max_sentences=None, max_tokens=None,
                        padding='pre', truncating='pre', value=0.):
    # Infer max lengths if needed.
    if max_sentences is None or max_tokens is None:
        max_sentences_computed = 0
        max_tokens_computed = 0
        for sent_seq in sequences:
            max_sentences_computed = max(max_sentences_computed, len(sent_seq))
            max_tokens_computed = max(max_tokens_computed, np.max([len(token_seq) for token_seq in sent_seq]))

        # Only use inferred values for None.
        max_sentences = min(max_sentences, max_sentences_computed)
        max_tokens = min(max_tokens, max_tokens_computed)

    result = np.ones(shape=(len(sequences), max_sentences, max_tokens)) * value

    for idx, sent_seq in enumerate(sequences):
        # empty list/array was found
        if not len(sent_seq):
            continue
        if truncating == 'pre':
            trunc = sent_seq[-max_sentences:]
        elif truncating == 'post':
            trunc = sent_seq[:max_sentences]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # Apply padding.
        if padding == 'post':
            result[idx, :len(trunc)] = _pad_token_sequences(trunc, max_tokens, padding, truncating, value)
        elif padding == 'pre':
            result[idx, -len(trunc):] = _pad_token_sequences(trunc, max_tokens, padding, truncating, value)
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return result


def pad_sequences(sequences, max_sentences=None, max_tokens=None,
                  padding='pre', truncating='post', value=0.):
    """Pads each sequence to the same length (length of the longest sequence or provided override).
    Args:
        sequences: list of list (samples, words) or list of list of list (samples, sentences, words)
        max_sentences: The max sentence length to use. If None, largest sentence length is used.
        max_tokens: The max word length to use. If None, largest word length is used.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than max_sentences or max_tokens
            either in the beginning or in the end of the sentence or word sequence respectively.
        value: The padding value.
    Returns:
        Numpy array of (samples, max_sentences, max_tokens) or (samples, max_tokens) depending on the sequence input.
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`.
    """

    # Determine if input is (samples, max_sentences, max_tokens) or not.
    if isinstance(sequences[0][0], list):
        x = _pad_sent_sequences(sequences, max_sentences, max_tokens, padding, truncating, value)
    else:
        x = _pad_token_sequences(sequences, max_tokens, padding, truncating, value)
    return np.array(x, dtype='int32')