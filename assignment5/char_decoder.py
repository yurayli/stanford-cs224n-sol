#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        pad_token_idx = target_vocab.char2id['<pad>']
        char_vocab_size = len(target_vocab.char2id)

        self.target_vocab = target_vocab
        self.decoderCharEmb = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=pad_token_idx)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, char_vocab_size, bias=False)
        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)   # (len, b, e)
        dec_hiddens, (last_hidden, last_cell) = self.charDecoder(X, dec_hidden)   # (len, b, h), (1, b, h), (1, b, h)
        scores = self.char_output_projection(dec_hiddens)   # (len, b, v)
        dec_hidden = (last_hidden, last_cell)
        return scores, dec_hidden
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # forward pass char_sequence
        logits, _ = self.forward(char_sequence[:-1], dec_hidden)   # (len-1, b, v), ((1, b, h), (1, b, h))

        # Zero out, probabilities for which we have nothing in the target char
        masks = (char_sequence != self.target_vocab.char2id['<pad>']).float()

        # cross-entropy loss
        P = F.log_softmax(logits, dim=-1)   # (len-1, b, v)
        # P = P.view(-1, P.size(-1))
        # chars_log_prob = P[np.arange(P.size(0)), char_sequence[1:].view(-1)]
        chars_log_prob = torch.gather(P, index=char_sequence[1:].unsqueeze(-1), dim=-1).squeeze(-1) * masks[1:]   # (len-1, b)

        return chars_log_prob.sum()
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        bch_size = initialStates[0].size(1)
        chars = np.zeros((bch_size, max_length), dtype=np.int32)   # (b, w_len)
        chars[:, 0] = np.ones(bch_size) * self.target_vocab.start_of_word

        inpts = torch.LongTensor(chars[:, 0], device=device)   # (b, )
        states = initialStates

        for t in range(1, max_length):
            char_logits, states = self.forward(inpts.unsqueeze(0), states)   # (1, b, v), (1, b, h), (1, b, h)
            inpts = torch.argmax(F.softmax(char_logits.squeeze(0), dim=-1), dim=1)   # (b, )
            chars[:, t] = inpts.data.cpu().numpy()

        decodedWords = []
        for i in range(bch_size):
            decodedWords.append(
                ''.join([self.target_vocab.id2char[cid] for cid in chars[i,1:] if cid>3]))

        return decodedWords
        ### END YOUR CODE

