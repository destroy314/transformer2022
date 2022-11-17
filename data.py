from conf import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from typing import Iterable, List

TRANS_PAIR = ("en", "de")
# TRANS_PAIR = ("de", "en")

# Place-holders
token_transform = {}
vocab_transform = {}

tok = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
# Create source and target language tokenizer. Make sure to install the dependencies.
token_transform[0] = get_tokenizer("spacy", tok[TRANS_PAIR[0]])
token_transform[1] = get_tokenizer("spacy", tok[TRANS_PAIR[1]])


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: int):
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

src_pad_idx, trg_pad_idx, trg_sos_idx = 1, 1, 2

for ln in [0, 1]:
    # Training data Iterator
    train_data = Multi30k(root="data", split="train", language_pair=TRANS_PAIR)
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_data, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )


# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [0, 1]:
    vocab_transform[ln].set_default_index(UNK_IDX)


enc_voc_size = len(vocab_transform[0])
dec_voc_size = len(vocab_transform[1])

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [0, 1]:
    text_transform[ln] = sequential_transforms(
        token_transform[ln],  # Tokenization
        vocab_transform[ln],  # Numericalization
        tensor_transform,
    )  # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[0](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[1](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


train_data = Multi30k(root="data", split="train", language_pair=TRANS_PAIR)
train_iter = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)


valid_data = Multi30k(root="data", split="train", language_pair=TRANS_PAIR)
valid_iter = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
