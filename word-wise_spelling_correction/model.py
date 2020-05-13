import argparse

import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, FC, to_variable, Layer, guard
from transformer import WrapEncoderLayer
import numpy as np
import random
import json
import paddle
from copy import copy
from datasetUtil import readVocab, loadNoisyInputLabel, all_sentence2examples, \
    wordReplacement, charReplacement, excludeSentenceWithUnknownWords
import sentencepiece as spm

class Params(object):
    def __init__(self, config_path):
        toFloat = set([
                "learning_rate", "beta1", "beta2", "eps", "label_smooth_eps",
                "prepostprocess_dropout", "attention_dropout", "relu_dropout"
            ])
        toBool = set(["oov", "weight_sharing", "test", "dynamic"])
        with open(config_path) as f:
            configs = json.load(f)
        for k,v in configs.items():
            if k in toFloat:
                setattr(self, k, float(v))
            elif k in toBool:
                setattr(self, k, bool(v))
            elif "path" in k and v is not None:
                setattr(self, k, configs["prefix"]+v)
            else:
                try:
                    setattr(self, k, int(v))
                except:
                    setattr(self, k, v)

def create_data(np_values, is_static=False):
    """
    create_data
    :param np_values:
    :param is_static:
    :return:
    """
    # pdb.set_trace()
    [
        src_word_np, src_pos_np, src_slf_attn_bias_np, src_bpe_np, src_bpe_pos_np, src_bpe_slf_attn_bias_np, \
            src_sentence_np, src_sentence_pos_np, src_sentence_slf_attn_bias_np, positional_idx, lbl_word_np
    ] = np_values

    if is_static:
        return [
            src_word_np, src_pos_np, src_slf_attn_bias_np, src_bpe_np, src_bpe_pos_np, src_bpe_slf_attn_bias_np, \
            src_sentence_np, src_sentence_pos_np, src_sentence_slf_attn_bias_np, positional_idx, lbl_word_np
        ]
    else:
        word_enc_inputs = [
            to_variable(src_sentence_np, name='src_sentence'), 
            to_variable(src_sentence_pos_np, name='src_sentence_pos'), 
            to_variable(src_sentence_slf_attn_bias_np, name='src_sentence_slf_attn_bias'),
        ]

        bpe_enc_inputs = [
            to_variable(src_bpe_np, name='src_bpe'),
            to_variable(src_bpe_pos_np, name='src_bpe_pos'),
            to_variable(src_bpe_slf_attn_bias_np, name='src_bpe_slf_attn_bias')
        ]

        char_enc_inputs = [
            to_variable(src_word_np, name='src_word'), 
            to_variable(src_pos_np, name='src_pos'), 
            to_variable(src_slf_attn_bias_np, name='src_slf_attn_bias'),
        ]

        positional_idx = to_variable(positional_idx, name='positional_idx')
        if lbl_word_np is not None:
            label = to_variable(lbl_word_np, name='lbl_word')
        else:
            label = None
        return word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label

def prepare_batch_input(insts, src_pad_idx, n_head):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """

    def __pad_batch_data(insts,
                         pad_idx,
                         n_head,
                         is_target=False,
                         is_label=False,
                         return_attn_bias=True,
                         return_max_len=True,
                         return_num_token=False):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        # Any token included in dict can be used to pad, since the paddings' loss
        # will be masked out by weights and make no effect on parameter gradients.
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if is_label:  # label weight
            inst_weight = np.array([[1.] * len(inst) + [0.] *
                                    (max_len - len(inst)) for inst in insts])
            return_list += [inst_weight.astype("float32").reshape([-1, 1])]
        else:  # position data
            inst_pos = np.array([
                list(range(0, len(inst))) + [0] * (max_len - len(inst))
                for inst in insts
            ])
            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones(
                    (inst_data.shape[0], max_len, max_len))
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape([-1, 1, max_len, max_len])
                slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                             [1, n_head, 1, 1]) * [-1e9]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                               (max_len - len(inst))
                                               for inst in insts])
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1])
            return_list += [slf_attn_bias_data.astype("float32")]
        if return_max_len:
            return_list += [max_len]
        if return_num_token:
            num_token = 0
            for inst in insts:
                num_token += len(inst)
            return_list += [num_token]
        return return_list if len(return_list) > 1 else return_list[0]

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[2] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    src_bpe, src_bpe_pos, src_bpe_slf_attn_bias, src_bpe_max_len = __pad_batch_data(
    [inst[1] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_bpe_word = src_bpe.reshape(-1, src_bpe_max_len, 1)
    src_bpe_pos = src_bpe_pos.reshape(-1, src_bpe_max_len, 1)

    src_sentence, src_sentence_pos, src_sentence_slf_attn_bias, src_sentence_max_len = __pad_batch_data(
    [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_sentence = src_sentence.reshape(-1, src_sentence_max_len, 1)
    src_sentence_pos = src_sentence_pos.reshape(-1, src_sentence_max_len, 1)

    positional_index = np.array([[i,inst[-2]] for i, inst in enumerate(insts)])
    if insts[0][-1] is not None:
        lbl_word = np.array([inst[-1] for inst in insts]).reshape(-1,1)
    else:
        lbl_word = None
    return [
        src_word, src_pos, src_slf_attn_bias, src_bpe_word, src_bpe_pos, src_bpe_slf_attn_bias, src_sentence,
        src_sentence_pos, src_sentence_slf_attn_bias, positional_index, lbl_word
    ]

def dynamicReaderCreator(input_path, word_replacement, char_vocab, word_vocab, bpe_model, oov=True, mode = "natural"):
    print("Using", mode, "mode!")
    dataset = []
    with open(input_path,"r") as f:
        for line in f:
            tokens_1B = line.strip().split()
            dataset.append(tokens_1B)

    input_size = len(dataset)
    block_size = input_size // params.n_split

    if not oov:
        dataset, _ = excludeSentenceWithUnknownWords(word_vocab, dataset)

    def reader():
        for j in range(params.n_split):
            if mode == "mix":
                if oov:
                    replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = word_vocab, sigma=0.1)
                    char_replaced_dataset, char_gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = word_vocab, sigma=0.1)
                else:
                    replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
                    char_replaced_dataset, char_gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
                replaced_dataset += char_replaced_dataset
                gold_labels += char_gold_labels

            elif mode == "char":
                if oov:
                    replaced_dataset, gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = word_vocab, sigma=0.1)
                else:
                    replaced_dataset, gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
            
            else: # natural
                if oov:
                    replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = word_vocab, sigma=0.1)
                else:
                    replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)

            all_examples, all_labels = all_sentence2examples(replaced_dataset, gold_labels)
            assert( len(all_examples) == len(all_labels))
            indices = [i for i in range(len(all_examples))]
            random.shuffle(indices)
            for i in indices:
                tokens, mistake, positional_idx = all_examples[i]
                tokens_idx = [word_vocab[token] for token in tokens]
                bpe_idx = bpe_model.EncodeAsIds(mistake)
                mistake_idx = [char_vocab.get(char,0) for char in mistake]
                label_idx = word_vocab[all_labels[i]]
                yield tokens_idx, bpe_idx, mistake_idx, positional_idx, label_idx

    return reader

def staticReaderCreator(input_path, label_path, char_vocab, word_vocab, bpe_model, train = True):

    def reader():
        with open(input_path, "r") as f:
            with open(label_path, "r") as lf:
                inputs = []
                labels = []
                count = 0
                for line, label_line in zip(f, lf):
                    count += 1
                    tokens = line.strip().split()
                    label = eval(label_line.strip())
                    label = {int(k):v for k, v in label.items()}
                    inputs.append(tokens)
                    labels.append(label)

                    if count >= params.limit:
                        all_examples, all_labels = all_sentence2examples(inputs,labels) 
                        assert( len(all_examples) == len(all_labels))
                        indices = [i for i in range(len(all_examples))]
                        random.shuffle(indices)

                        for i in indices:
                            tokens, mistake, positional_idx = all_examples[i]
                            if train:
                                label_idx = word_vocab.get(all_labels[i],params.oov_idx)
                            else:
                                label_idx = None
                            tokens_idx = [word_vocab.get(token, params.unk_idx) for token in tokens]
                            bpe_idx = bpe_model.EncodeAsIds(mistake)
                            mistake_idx = [char_vocab.get(char,0) for char in mistake] # For extremely rare case of char OOV, just assign index 0
                            yield (tokens_idx, bpe_idx, mistake_idx, positional_idx, label_idx)

                        inputs = []
                        labels = []
                        count = 0

                all_examples, all_labels = all_sentence2examples(inputs,labels)
                assert( len(all_examples) == len(all_labels))
                indices = [i for i in range(len(all_examples))]
                random.shuffle(indices)

                for i in indices:
                    tokens, mistake, positional_idx = all_examples[i]
                    if train:
                        label_idx = word_vocab.get(all_labels[i],params.oov_idx)
                    else:
                        label_idx = None
                    tokens_idx = [word_vocab.get(token, params.unk_idx) for token in tokens]
                    bpe_idx = bpe_model.EncodeAsIds(mistake)
                    mistake_idx = [char_vocab.get(char,0) for char in mistake] # For extremely rare case of char OOV, just sign index 0
                    yield (tokens_idx, bpe_idx, mistake_idx, positional_idx, label_idx)
    return reader, None

def smallStaticReaderCreator(input_path, label_path, char_vocab, word_vocab, bpe_model, train = True):
    inputs = []
    with open(input_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            inputs.append(tokens)

    labels = loadNoisyInputLabel(label_path)
    input_size = len(inputs)
    block_size = input_size // params.n_split

    if not train:
        global all_examples, all_labels
        all_examples, all_labels = all_sentence2examples(inputs, labels)

    def text_reader():
        global all_examples, all_labels
        for i in range(len(all_examples)):
            tokens, mistake, idx = all_examples[i]
            yield (tokens, mistake, idx, all_labels[i])

    def reader():
        global all_examples, all_labels
        if train:
            for j in range(params.n_split):
                all_examples, all_labels = all_sentence2examples(inputs[j * block_size : (j+1) * block_size],\
                     labels[j * block_size : (j+1) * block_size])

                assert( len(all_examples) == len(all_labels))
                indices = [i for i in range(len(all_examples))]
                random.shuffle(indices)

                for i in indices:
                    tokens, mistake, positional_idx = all_examples[i]
                    if train:
                        label_idx = word_vocab.get(all_labels[i],params.oov_idx)
                    else:
                        label_idx = None
                    tokens_idx = [word_vocab.get(token, params.unk_idx) for token in tokens]
                    bpe_idx = bpe_model.EncodeAsIds(mistake)
                    mistake_idx = [char_vocab.get(char,0) for char in mistake] # For extremely rare case of char OOV, just sign index 0
                    yield (tokens_idx, bpe_idx, mistake_idx, positional_idx, label_idx)
        else:
            for i in range(len(all_examples)):
                tokens, mistake, positional_idx = all_examples[i]
                if train:
                    label_idx = word_vocab.get(all_labels[i],params.oov_idx)
                else:
                    label_idx = None
                tokens_idx = [word_vocab.get(token, params.unk_idx) for token in tokens]
                bpe_idx = bpe_model.EncodeAsIds(mistake)
                mistake_idx = [char_vocab.get(char,0) for char in mistake] # For extremely rare case of char OOV, just sign index 0
                yield (tokens_idx, bpe_idx, mistake_idx, positional_idx, label_idx)
    
    return reader, text_reader

def inferenceReaderCreator(input_path, char_vocab, word_vocab, bpe_model, label_path = None, detection_path=None):
    """
    Directly return a batch. A batch is a sentence with different positions to predict.
    """
    dataset = []
    with open(input_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            dataset.append(tokens)

    if label_path is not None:
        labels = loadNoisyInputLabel(label_path)
        assert(len(dataset) == len(labels))

    if detection_path is not None:
        all_detections = []
        with open(detection_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                all_detections.append(tokens)

    def reader():
        for i in range(len(dataset)):
            sentence_input = []
            tokens = dataset[i]
            tokens_idx = [word_vocab.get(token, params.unk_idx) for token in tokens]
            if detection_path is not None:
                detections = all_detections[i]
                for j, (token, detection) in enumerate(zip(tokens, detections)):
                    if token not in word_vocab or detection == "M":
                        mistake_idx = [char_vocab.get(char,0) for char in token] # For extremely rare case of char OOV, just sign index 0
                        bpe_idx = bpe_model.EncodeAsIds(token)
                        sentence_input.append((tokens_idx, bpe_idx, mistake_idx, j, None))
            else:
                for j, token in enumerate(tokens):
                    if token not in word_vocab:
                        mistake_idx = [char_vocab.get(char,0) for char in token] # For extremely rare case of char OOV, just sign index 0
                        bpe_idx = bpe_model.EncodeAsIds(token)
                        sentence_input.append((tokens_idx, bpe_idx, mistake_idx, j, None))
            yield sentence_input

    def text_reader():
        for i in range(len(dataset)):
            tokens = dataset[i]
            if label_path is not None:
                label = labels[i]
            else:
                label = None
            yield tokens, label

    return reader, text_reader

class CharactorSpellingCorrector(Layer):
    """
    model
    """

    def __init__(self, name_scope, src_vocab_size, trg_vocab_size, max_length,
                 n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd, weight_sharing,
                 label_smooth_eps):
        super(CharactorSpellingCorrector, self).__init__(name_scope)
        self._label_smooth_eps = label_smooth_eps
        self._trg_vocab_size = trg_vocab_size
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
        self._wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), src_vocab_size, max_length, n_layer, n_head,
            d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._fc = FC(self.full_name(),
              size=self._trg_vocab_size,
              bias_attr=False)

    def forward(self, enc_inputs, label):
        """
        forward
        :param enc_inputs:
        :param dec_inputs:
        :param label:
        :param weights:
        :return:
        """
        enc_output = self._wrap_encoder_layer(enc_inputs)
        enc_output_reshape = enc_output[:,0,:] # Just take the first char representation
        predict = self._fc(enc_output_reshape)
        if label is not None:
            if self._label_smooth_eps:
                label_out = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size),
                    epsilon=self._label_smooth_eps)
            else:
                label_out = fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size)
            cost = fluid.layers.softmax_with_cross_entropy(
                logits=predict,
                label=label_out,
                soft_label=True if self._label_smooth_eps else False)
            sum_cost = fluid.layers.reduce_sum(cost)
            token_num = fluid.layers.reduce_sum(cost*0+1)
            token_num.stop_gradient = True
            avg_cost = sum_cost / token_num
            return avg_cost, predict

        else:
            return predict

class SpellingCorrector(Layer):
    """
    model
    """

    def __init__(self, name_scope, src_vocab_size, char_vocab_size, trg_vocab_size, word_max_length,
                 sentence_max_length, n_layer, n_char_layer, n_head, d_key, d_value, 
                 d_model, d_char_model, d_inner_hid, d_char_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd, weight_sharing,
                 label_smooth_eps):
        super(SpellingCorrector, self).__init__(name_scope)
        self._label_smooth_eps = label_smooth_eps
        self._trg_vocab_size = trg_vocab_size
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )

        self._char_wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), char_vocab_size, word_max_length, n_char_layer, n_head,
            d_key, d_value, d_char_model, d_char_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._word_wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), src_vocab_size, sentence_max_length, n_layer, n_head,
            d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._fc = FC(self.full_name(),
              size=self._trg_vocab_size,
              bias_attr=False)

    def forward(self, word_enc_inputs, char_enc_inputs, positional_idx, label):
        """
        forward
        :param enc_inputs:
        :param dec_inputs:
        :param label:
        :param weights:
        :return:
        """
        char_enc_output = self._char_wrap_encoder_layer(char_enc_inputs)
        char_enc_output_reshape = char_enc_output[:,0,:] # Just take the first char representation

        word_enc_output = self._word_wrap_encoder_layer(word_enc_inputs)
        word_enc_output_reshape = fluid.layers.gather_nd(word_enc_output, positional_idx)

        enc_output = fluid.layers.concat([word_enc_output_reshape, char_enc_output_reshape],-1)

        predict = self._fc(enc_output)
        if label is not None:
            if self._label_smooth_eps:
                label_out = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size),
                    epsilon=self._label_smooth_eps)
            else:
                label_out = fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size)
            cost = fluid.layers.softmax_with_cross_entropy(
                logits=predict,
                label=label_out,
                soft_label=True if self._label_smooth_eps else False)
            sum_cost = fluid.layers.reduce_sum(cost)
            token_num = fluid.layers.reduce_sum(cost*0+1) # Always 1 for now.
            token_num.stop_gradient = True
            avg_cost = sum_cost / token_num
            return avg_cost, predict
        else:
            return predict

class FullSpellingCorrector(Layer):
    """
    model
    """

    def __init__(self, name_scope, src_vocab_size, bpe_vocab_size, char_vocab_size, trg_vocab_size, word_max_length,
                 sentence_max_length, n_layer, n_char_layer, n_head, d_key, d_value, 
                 d_model, d_char_model, d_inner_hid, d_char_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd, weight_sharing,
                 label_smooth_eps):
        super(FullSpellingCorrector, self).__init__(name_scope)
        self._label_smooth_eps = label_smooth_eps
        self._trg_vocab_size = trg_vocab_size
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )

        self._char_wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), char_vocab_size, word_max_length, n_char_layer, n_head,
            d_key, d_value, d_char_model, d_char_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._bpe_wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), bpe_vocab_size, word_max_length, n_char_layer, n_head,
            d_key, d_value, d_char_model, d_char_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._word_wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), src_vocab_size, sentence_max_length, n_layer, n_head,
            d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        self._fc = FC(self.full_name(),
              size=self._trg_vocab_size,
              bias_attr=False)

    def forward(self, word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label):
        """
        forward
        :param enc_inputs:
        :param dec_inputs:
        :param label:
        :param weights:
        :return:
        """
        char_enc_output = self._char_wrap_encoder_layer(char_enc_inputs)
        char_enc_output_reshape = char_enc_output[:,0,:] # Just take the first char representation

        bpe_enc_output = self._bpe_wrap_encoder_layer(bpe_enc_inputs)
        bpe_enc_output_reshape = bpe_enc_output[:,0,:] # Just take the first bpe representation

        word_enc_output = self._word_wrap_encoder_layer(word_enc_inputs)
        word_enc_output_reshape = fluid.layers.gather_nd(word_enc_output, positional_idx)

        enc_output = fluid.layers.concat([word_enc_output_reshape, bpe_enc_output_reshape, char_enc_output_reshape],-1)

        predict = self._fc(enc_output)
        if label is not None:
            if self._label_smooth_eps:
                label_out = fluid.layers.label_smooth(
                    label=fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size),
                    epsilon=self._label_smooth_eps)
            else:
                label_out = fluid.layers.one_hot(
                        input=label, depth=self._trg_vocab_size)
            cost = fluid.layers.softmax_with_cross_entropy(
                logits=predict,
                label=label_out,
                soft_label=True if self._label_smooth_eps else False)
            sum_cost = fluid.layers.reduce_sum(cost)
            token_num = fluid.layers.reduce_sum(cost*0+1) # Always 1 for now.
            token_num.stop_gradient = True
            avg_cost = sum_cost / token_num
            return avg_cost, predict
        else:
            return predict

def create_model():
    if params.model_type == "char_only":
        print("Using char model.")
        spelling_corrector = CharactorSpellingCorrector(
            'spelling_corrector', params.char_vocab_size,
            params.trg_vocab_size, params.word_max_length + 1,
            params.n_char_layer, params.n_head,
            params.d_key, params.d_value,
            params.d_char_model, params.d_char_inner_hid,
            params.prepostprocess_dropout,
            params.attention_dropout, params.relu_dropout,
            params.preprocess_cmd, params.postprocess_cmd,
            params.weight_sharing, params.label_smooth_eps)

    elif params.model_type == "standard":
        print("Using contextualized model.")
        spelling_corrector = SpellingCorrector(
            'spelling_corrector', params.src_vocab_size,  params.char_vocab_size,
            params.trg_vocab_size, params.word_max_length + 1, params.sentence_max_length + 1,
            params.n_layer, params.n_char_layer, params.n_head,
            params.d_key, params.d_value,
            params.d_model, params.d_char_model, 
            params.d_inner_hid, params.d_char_inner_hid,
            params.prepostprocess_dropout,
            params.attention_dropout, params.relu_dropout,
            params.preprocess_cmd, params.postprocess_cmd,
            params.weight_sharing, params.label_smooth_eps)

    else:
        print("Using contextualized model.")
        spelling_corrector = FullSpellingCorrector(
            'spelling_corrector', params.src_vocab_size, params.bpe_vocab_size, params.char_vocab_size,
            params.trg_vocab_size, params.word_max_length + 1, params.sentence_max_length + 1,
            params.n_layer, params.n_char_layer, params.n_head,
            params.d_key, params.d_value,
            params.d_model, params.d_char_model, 
            params.d_inner_hid, params.d_char_inner_hid,
            params.prepostprocess_dropout,
            params.attention_dropout, params.relu_dropout,
            params.preprocess_cmd, params.postprocess_cmd,
            params.weight_sharing, params.label_smooth_eps)
    return spelling_corrector

def test_misspeling(reader, model):
    correct = 0
    total = 0
    avg_loss_set = []
    for batch in reader():
        np_values = prepare_batch_input(
            batch, params.src_pad_idx,
            params.n_head)

        word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label = create_data(np_values)
        if params.model_type == "char_only":
            dy_avg_cost, dy_predict = model(char_enc_inputs, label)
        elif params.model_type == "standard":
            dy_avg_cost, dy_predict = model(word_enc_inputs, char_enc_inputs, positional_idx, label)
        else:
            dy_avg_cost, dy_predict = model(word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label)

        avg_loss_set.append(float(dy_avg_cost.numpy()))

        batch_prediction = np.argmax(dy_predict.numpy(), axis=1)
        label_np = label.numpy().reshape(-1)
        correct += np.sum(batch_prediction == label_np)
        total += label_np.size
    return correct / total, np.mean(avg_loss_set)

def train(dynamic_generation = True, model_path = None):

    with open(params.train_replacement) as f:
        trainWordReplacement = json.load(f)

    with guard():
        spelling_corrector = create_model()
        
        if model_path is not None:
            model_dict, _ = fluid.dygraph.load_dygraph(model_path)
            spelling_corrector.load_dict(model_dict)
            print("Model loaded.")

        lr = fluid.layers.learning_rate_scheduler.noam_decay(
               1/(params.warmup_steps *(params.learning_rate ** 2)),
               params.warmup_steps)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, beta1=params.beta1, beta2=params.beta2, 
                epsilon=params.eps)

        if dynamic_generation:
            train_reader = dynamicReaderCreator(params.train_input_path, trainWordReplacement, char_vocab, word_vocab, bpe_model, oov = params.oov, mode = params.mode)
        else:
            train_reader, _ = staticReaderCreator(params.train_input_path, params.train_label_path, char_vocab, word_vocab, bpe_model, train=True)
        train_reader = paddle.batch(train_reader, batch_size=params.batch_size)

        test_reader, _ = staticReaderCreator(params.dev_input_path, params.dev_label_path, char_vocab, word_vocab, bpe_model, train=True)
        test_reader = paddle.batch(test_reader, batch_size=params.batch_size)

        best_acc = 0
        for i in range(200):
            dy_step = 0
            avg_costs = []
            for batch in train_reader():
                np_values = prepare_batch_input(
                    batch, params.src_pad_idx,
                    params.n_head)

                word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label = create_data(np_values)
                if params.model_type == "char_only":
                    dy_avg_cost, dy_predict = spelling_corrector(
                        char_enc_inputs, label)

                elif params.model_type == "standard":
                    dy_avg_cost, dy_predict = spelling_corrector(
                        word_enc_inputs, char_enc_inputs, positional_idx, label)

                else:
                    dy_avg_cost, dy_predict = spelling_corrector(
                        word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label)
                dy_avg_cost.backward()
                optimizer.minimize(dy_avg_cost)
                spelling_corrector.clear_gradients()
                dy_step = dy_step + 1
                dy_avg_cost_float = float(dy_avg_cost.numpy())
                avg_costs.append(dy_avg_cost_float)

                if dy_step % 5 == 0:
                    print("pass num : {}, batch_id: {}, dy_graph avg loss: {}".
                        format(i, dy_step, dy_avg_cost_float))
                if dy_step % 5 == 0:
                    print("Periodically save the model.")
                    fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), params.periodic_checkpoint)
            print("Pass: {} finished. Avg training cost: {}".format(i, np.mean(avg_costs)))
            
            spelling_corrector.eval()
            test_acc, test_loss = test_misspeling(test_reader, spelling_corrector)
            spelling_corrector.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                i, test_loss, test_acc))
            if best_acc <= test_acc:
                fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), params.best_checkpoint)
                print("New best accuracy. New model saved.")
                best_acc = test_acc
            else:
                print("Accuracy not improved. Skip saving model.")

def train_parallel(dynamic_generation = True, model_path = None):

    with open(params.train_replacement) as f:
        trainWordReplacement = json.load(f)

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    with guard(place):
        strategy = fluid.dygraph.parallel.prepare_context()
        spelling_corrector = create_model()

        if model_path is not None:
            try:
                model_dict, _ = fluid.dygraph.load_dygraph(model_path)
                spelling_corrector.load_dict(model_dict)
                print(model_path, "loaded.")
            except:
                print("Model not found, training from scratch.")

        spelling_corrector = fluid.dygraph.parallel.DataParallel(spelling_corrector, strategy)

        lr = fluid.layers.learning_rate_scheduler.noam_decay(
               1/(params.warmup_steps *(params.learning_rate ** 2)),
               params.warmup_steps)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, beta1=params.beta1, beta2=params.beta2, 
                epsilon=params.eps)

        if dynamic_generation:
            train_reader = dynamicReaderCreator(params.train_input_path, trainWordReplacement, char_vocab, word_vocab, bpe_model, oov = params.oov, mode = params.mode)
        else:
            train_reader, _ = staticReaderCreator(params.train_input_path, params.train_label_path, char_vocab, word_vocab, bpe_model, train=True)
        train_reader = paddle.batch(train_reader, batch_size=params.batch_size)

        train_reader = fluid.contrib.reader.distributed_batch_reader(train_reader)

        test_reader, _ = staticReaderCreator(params.dev_input_path, params.dev_label_path, char_vocab, word_vocab, bpe_model, train=True)
        test_reader = paddle.batch(test_reader, batch_size=params.batch_size)

        best_acc = 0
        for i in range(200):
            dy_step = 0
            avg_costs = []
            for batch in train_reader():
                np_values = prepare_batch_input(
                    batch, params.src_pad_idx,
                    params.n_head)

                word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label = create_data(np_values)
                if params.model_type == "char_only":
                    dy_avg_cost, dy_predict = spelling_corrector(
                        char_enc_inputs, label)
                elif params.model_type == "standard":
                    dy_avg_cost, dy_predict = spelling_corrector(
                        word_enc_inputs, char_enc_inputs, positional_idx, label)
                else:
                    dy_avg_cost, dy_predict = spelling_corrector(
                        word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, label)
                
                dy_avg_cost = spelling_corrector.scale_loss(dy_avg_cost)
                dy_avg_cost.backward()
                spelling_corrector.apply_collective_grads()

                optimizer.minimize(dy_avg_cost)
                spelling_corrector.clear_gradients()
                dy_step = dy_step + 1
                dy_avg_cost_float = float(dy_avg_cost.numpy())
                avg_costs.append(dy_avg_cost_float)

                if dy_step % 100 == 0:
                    print("pass num : {}, batch_id: {}, dy_graph avg loss: {}".
                        format(i, dy_step, dy_avg_cost_float))
                if dy_step % 1000 == 0:
                    print("Periodically save the model.")
                    fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), params.periodic_checkpoint)

                    spelling_corrector.eval()
                    test_acc, test_loss = test_misspeling(test_reader, spelling_corrector)
                    spelling_corrector.train()
                    print("Loss at step {} , Test avg_loss is: {}, acc is: {}".format(
                        dy_step, test_loss, test_acc))
                    if best_acc <= test_acc:
                        fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), params.best_checkpoint)
                        print("New best accuracy. New model saved.")
                        best_acc = test_acc
                    else:
                        print("Accuracy not improved. Skip saving model.")

            print("Pass: {} finished. Avg training cost: {}".format(i, np.mean(avg_costs)))
            
            spelling_corrector.eval()
            test_acc, test_loss = test_misspeling(test_reader, spelling_corrector)
            spelling_corrector.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                i, test_loss, test_acc))
            if best_acc <= test_acc:
                fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), params.best_checkpoint)
                print("New best accuracy. New model saved.")
                best_acc = test_acc
            else:
                print("Accuracy not improved. Skip saving model.")

def debug_misspelling(model_path, input_path, label_path):
    with guard():
        spelling_corrector = create_model()

        model_dict, _ = fluid.dygraph.load_dygraph(model_path)
        spelling_corrector.load_dict(model_dict)
        print("Model loaded.")

        test_reader, _ = smallStaticReaderCreator(input_path, label_path, char_vocab, word_vocab, bpe_model, train=True)
        test_reader = paddle.batch(test_reader, batch_size=params.batch_size) 
            
        spelling_corrector.eval()
        test_acc, test_loss = test_misspeling(test_reader, spelling_corrector)
        print("Correction only accuracy:", test_acc, test_loss)

def inference_misspelling(model_path, input_path, iterative=True, label_path=None, detection_path = None):
    corrected_sentences = []
    original_sentences = []
    ground_truth_labels = []
    with guard():
        spelling_corrector = create_model()

        model_dict, _ = fluid.dygraph.load_dygraph(model_path)
        spelling_corrector.load_dict(model_dict)
        print("Model loaded.")

        inference_reader, text_reader = inferenceReaderCreator(
            input_path, char_vocab, word_vocab, bpe_model, 
            label_path=label_path, detection_path=detection_path
            )

        spelling_corrector.eval()
        for batch, texts in zip(inference_reader(), text_reader()):
            original_tokens, labels = texts
            tokens = original_tokens
            original_sentences.append(original_tokens)
            ground_truth_labels.append(labels)
            while len(batch) > 0:
                np_values = prepare_batch_input(
                    batch, params.src_pad_idx,
                    params.n_head)
                word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, _ = create_data(np_values)
                if params.model_type == "char_only":
                    dy_predict = spelling_corrector(
                        char_enc_inputs, None)
                elif params.model_type == "standard":
                    dy_predict = spelling_corrector(
                        word_enc_inputs, char_enc_inputs, positional_idx, None)
                else:
                    dy_predict = spelling_corrector(
                        word_enc_inputs, bpe_enc_inputs, char_enc_inputs, positional_idx, None)
                # batch_prediction.shape = (# of misspelling, vocab_size)
                if iterative:
                    batch_prediction = np.argmax(dy_predict.numpy(), axis=1)
                    # Choose the prediction with highest score
                    batch_prediction_score = np.max(fluid.layers.softmax(dy_predict).numpy(), axis=1) 
                    best_position_index = np.argmax(batch_prediction_score)

                    best_position = positional_idx.numpy()[best_position_index][1]
                    best_prediction = batch_prediction[best_position_index]
                    #print(best_prediction, best_position, tokens[best_position], reverse_word_vocab[best_prediction])
                    batch, tokens = correctToken(tokens, best_prediction, best_position)
                else:
                    batch_prediction = np.argmax(dy_predict.numpy(), axis=1)
                    positional_indices = positional_idx.numpy()[:,1]
                    batch, tokens = combineToken(tokens, batch_prediction, positional_indices)
            corrected_sentences.append(copyOOV(original_tokens, tokens))
    return corrected_sentences, original_sentences, ground_truth_labels

def correctToken(tokens, word_index, position):
    sentence_input = []
    corrected_tokens = copy(tokens)
    corrected_tokens[position] = reverse_word_vocab[word_index]
    tokens_idx = [word_vocab.get(token, params.unk_idx) for token in corrected_tokens]
    for i, token in enumerate(corrected_tokens):
        if token not in word_vocab:
            mistake_idx = [char_vocab[char] for char in token]
            bpe_idx = bpe_model.EncodeAsIds(token)
            sentence_input.append((tokens_idx, bpe_idx, mistake_idx, i, None))
    return sentence_input, corrected_tokens

def combineToken(tokens, batch_prediction, positional_indices):
    assert(batch_prediction.shape == positional_indices.shape)
    corrected_tokens = copy(tokens)
    for prediction_index, position in zip(batch_prediction, positional_indices):
        corrected_tokens[position] = reverse_word_vocab[prediction_index]
    return [], corrected_tokens

def copyOOV(original_tokens, predicted_tokens):
    processed_tokens = []
    for original_token, predicted_token in zip(original_tokens, predicted_tokens):
        if predicted_token == "<oov>":
            processed_tokens.append(original_token)
        else:
            processed_tokens.append(predicted_token)
    return processed_tokens

def evaluation(corrected_sentences, original_sentences, ground_truth_labels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for tokens, prediction, labels in zip(original_sentences, corrected_sentences, ground_truth_labels):
        correct_tokens = copy(tokens)
        for i, gold in labels.items():
            correct_tokens[i] = gold
        for token, gold, pred in zip(tokens, correct_tokens, prediction):
            if token!=pred:
                if pred==gold:
                    TP += 1
                else:
                    FP += 1
            else:
                if pred==gold:
                    TN += 1
                else:
                    FN += 1
    print("Precision:",TP / (TP+FP))
    print("Recall:",TP / (TP+FN))
    print("Accuracy:", (TP + TN)/(TP+FP+FN+TN))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="JSON config file path")
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    params = Params(args.config_file)

    print("Model name:", params.model_name)
    char_vocab, word_vocab = readVocab(params.char_path, params.word_path)
    reverse_word_vocab = {index: word for word, index in word_vocab.items()}

    bpe_model = spm.SentencePieceProcessor()
    bpe_model.Load(params.bpe_path)

    if not args.test:
        #train(dynamic_generation = False, model_path = None)
        train_parallel(dynamic_generation = params.dynamic, model_path = params.best_checkpoint)
    else:
        print("Evaluating dev set...")
        debug_misspelling(params.best_checkpoint, params.dev_input_path,params.dev_label_path)
        print("Vanilla detection:")
        corrected_sentences, original_sentences, ground_truth_labels = inference_misspelling(
            params.best_checkpoint,
            params.dev_input_path,
            iterative = False, 
            label_path = params.dev_label_path, 
            detection_path = None
            )
        #for tokens in corrected_sentences:
        #    print(" ".join(tokens))
        evaluation(corrected_sentences, original_sentences, ground_truth_labels)

        if params.dev_detection_path is not None:
            print("ERNIE detection")
            corrected_sentences, original_sentences, ground_truth_labels = inference_misspelling(
                params.best_checkpoint,
                params.dev_input_path,
                iterative = False, 
                label_path = params.dev_label_path, 
                detection_path = params.dev_detection_path
                )
            #for tokens in corrected_sentences:
            #    print(" ".join(tokens))
            evaluation(corrected_sentences, original_sentences, ground_truth_labels)

        print("Evaluating test set...")
        debug_misspelling(params.best_checkpoint, params.test_input_path,params.test_label_path)
        print("Vanilla detection:")
        corrected_sentences, original_sentences, ground_truth_labels = inference_misspelling(
            params.best_checkpoint,
            params.test_input_path,
            iterative = False, 
            label_path = params.test_label_path, 
            detection_path = None
            )
        #for tokens in corrected_sentences:
        #    print(" ".join(tokens))
        evaluation(corrected_sentences, original_sentences, ground_truth_labels)

        if params.test_detection_path is not None:
            print("ERNIE detection:")
            corrected_sentences, original_sentences, ground_truth_labels = inference_misspelling(
                params.best_checkpoint,
                params.test_input_path,
                iterative = False, 
                label_path = params.test_label_path, 
                detection_path = params.test_detection_path
                )
            #for tokens in corrected_sentences:
            #    print(" ".join(tokens))
            evaluation(corrected_sentences, original_sentences, ground_truth_labels)