import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, FC, to_variable, Layer, guard
import numpy as np

class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, name_scope, process_cmd, shape_len=None):
        super(PrePostProcessLayer, self).__init__(name_scope)
        for cmd in process_cmd:
            if cmd == "n":
                self._layer_norm = LayerNorm(
                    name_scope=self.full_name(),
                    begin_norm_axis=shape_len - 1,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(1.)),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(0.)))

    def forward(self, prev_out, out, process_cmd, dropout_rate=0.):
        """
        forward
        :param prev_out:
        :param out:
        :param process_cmd:
        :param dropout_rate:
        :return:
        """
        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out else out
            elif cmd == "n":  # add layer normalization
                out = self._layer_norm(out)
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = fluid.layers.dropout(
                        out,
                        dropout_prob=dropout_rate,
                        #seed=ModelHyperParams.dropout_seed,
                        is_test=False)
        return out

class PositionwiseFeedForwardLayer(Layer):
    """
    PositionwiseFeedForwardLayer
    """

    def __init__(self, name_scope, d_inner_hid, d_hid, dropout_rate):
        super(PositionwiseFeedForwardLayer, self).__init__(name_scope)
        self._i2h = FC(name_scope=self.full_name(),
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
        self._h2o = FC(name_scope=self.full_name(),
                       size=d_hid,
                       num_flatten_dims=2)
        self._dropout_rate = dropout_rate

    def forward(self, x):
        """
        forward
        :param x:
        :return:
        """
        hidden = self._i2h(x)
        if self._dropout_rate:
            hidden = fluid.layers.dropout(
                hidden,
                dropout_prob=self._dropout_rate,
                #seed=ModelHyperParams.dropout_seed,
                is_test=False)
        out = self._h2o(hidden)
        return out

class MultiHeadAttentionLayer(Layer):
    """
    MultiHeadAttentionLayer
    """

    def __init__(self,
                 name_scope,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 cache=None,
                 gather_idx=None,
                 static_kv=False):
        super(MultiHeadAttentionLayer, self).__init__(name_scope)
        self._n_head = n_head
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._dropout_rate = dropout_rate
        self._q_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._k_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._v_fc = FC(name_scope=self.full_name(),
                        size=d_value * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._proj_fc = FC(name_scope=self.full_name(),
                           size=self._d_model,
                           bias_attr=False,
                           num_flatten_dims=2)

    def forward(self, queries, keys, values, attn_bias):
        """
        forward
        :param queries:
        :param keys:
        :param values:
        :param attn_bias:
        :return:
        """
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._q_fc(queries)
        k = self._k_fc(keys)
        v = self._v_fc(values)

        # split head
        reshaped_q = fluid.layers.reshape(
            x=q, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_q = fluid.layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        reshaped_k = fluid.layers.reshape(
            x=k, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_k = fluid.layers.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = fluid.layers.reshape(
            x=v, shape=[0, 0, self._n_head, self._d_value], inplace=False)
        transpose_v = fluid.layers.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

        # scale dot product attention
        product = fluid.layers.matmul(
            x=transpose_q,
            y=transpose_k,
            transpose_y=True,
            alpha=self._d_model**-0.5)
        if attn_bias:
            product += attn_bias
        weights = fluid.layers.softmax(product)
        if self._dropout_rate:
            weights_droped = fluid.layers.dropout(
                weights,
                dropout_prob=self._dropout_rate,
                #seed=ModelHyperParams.dropout_seed,
                is_test=False)
            out = fluid.layers.matmul(weights_droped, transpose_v)
        else:
            out = fluid.layers.matmul(weights, transpose_v)

        # combine heads
        if len(out.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
        final_out = fluid.layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=False)

        # fc to output
        proj_out = self._proj_fc(final_out)
        return proj_out

class EncoderSubLayer(Layer):
    """
    EncoderSubLayer
    """

    def __init__(self,
                 name_scope,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderSubLayer, self).__init__(name_scope)
        self._preprocess_cmd = preprocess_cmd
        self._postprocess_cmd = postprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout

        self._preprocess_layer = PrePostProcessLayer(self.full_name(),
                                                     self._preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(
            self.full_name(), d_key, d_value, d_model, n_head,
            attention_dropout)
        self._postprocess_layer = PrePostProcessLayer(
            self.full_name(), self._postprocess_cmd, None)
        self._preprocess_layer2 = PrePostProcessLayer(self.full_name(),
                                                      self._preprocess_cmd, 3)
        self._positionwise_feed_forward = PositionwiseFeedForwardLayer(
            self.full_name(), d_inner_hid, d_model, relu_dropout)
        self._postprocess_layer2 = PrePostProcessLayer(
            self.full_name(), self._postprocess_cmd, None)

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        pre_process_multihead = self._preprocess_layer(
            None, enc_input, self._preprocess_cmd, self._prepostprocess_dropout)
        attn_output = self._multihead_attention_layer(pre_process_multihead,
                                                      None, None, attn_bias)
        attn_output = self._postprocess_layer(enc_input, attn_output,
                                              self._postprocess_cmd,
                                              self._prepostprocess_dropout)
        pre_process2_output = self._preprocess_layer2(
            None, attn_output, self._preprocess_cmd,
            self._prepostprocess_dropout)
        ffd_output = self._positionwise_feed_forward(pre_process2_output)
        return self._postprocess_layer2(attn_output, ffd_output,
                                        self._postprocess_cmd,
                                        self._prepostprocess_dropout)


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self,
                 name_scope,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderLayer, self).__init__(name_scope)
        self._preprocess_cmd = preprocess_cmd
        self._encoder_sublayers = list()
        self._prepostprocess_dropout = prepostprocess_dropout
        self._n_layer = n_layer
        self._preprocess_layer = PrePostProcessLayer(self.full_name(),
                                                     self._preprocess_cmd, 3)
        for i in range(n_layer):
            self._encoder_sublayers.append(
                self.add_sublayer(
                    'esl_%d' % i,
                    EncoderSubLayer(
                        self.full_name(), n_head, d_key, d_value, d_model,
                        d_inner_hid, prepostprocess_dropout, attention_dropout,
                        relu_dropout, preprocess_cmd, postprocess_cmd)))

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        for i in range(self._n_layer):
            enc_output = self._encoder_sublayers[i](enc_input, attn_bias)
            enc_input = enc_output

        return self._preprocess_layer(None, enc_output, self._preprocess_cmd,
                                      self._prepostprocess_dropout)


class PrepareEncoderDecoderLayer(Layer):
    """
    PrepareEncoderDecoderLayer
    """

    def __init__(self,
                 name_scope,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareEncoderDecoderLayer, self).__init__(name_scope)
        self._src_max_len = src_max_len
        self._src_emb_dim = src_emb_dim
        self._src_vocab_size = src_vocab_size
        self._dropout_rate = dropout_rate
        self._input_emb = Embedding(
            name_scope=self.full_name(),
            size=[src_vocab_size, src_emb_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Normal(0., src_emb_dim**-0.5)))

        pos_inp = self.position_encoding_init(self._src_max_len + 1,
                                self._src_emb_dim)
        self._pos_emb = Embedding(
            name_scope=self.full_name(),
            size=[self._src_max_len, src_emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(pos_inp),
                trainable=False))
        # use in dygraph_mode to fit different length batch
        # self._pos_emb._w = to_variable(
        #     position_encoding_init(self._src_max_len, self._src_emb_dim))

    def position_encoding_init(self, n_position, d_pos_vec):
        """
        Generate the initial values for the sinusoid position encoding table.
        """
        channels = d_pos_vec
        position = np.arange(n_position)
        num_timescales = channels // 2
        log_timescale_increment = (np.log(float(1e4) / float(1)) /
                                (num_timescales - 1))
        inv_timescales = np.exp(np.arange(
            num_timescales)) * -log_timescale_increment
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                                0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
        position_enc = signal
        return position_enc.astype("float32")

    def forward(self, src_word, src_pos):
        """
        forward
        :param src_word:
        :param src_pos:
        :return:
        """
        # print(self._input_emb._w._numpy().shape)
        src_word_emb = self._input_emb(src_word)
        src_word_emb = fluid.layers.scale(
            x=src_word_emb, scale=self._src_emb_dim**0.5)
        # # TODO change this to fit dynamic length input
        src_pos_emb = self._pos_emb(src_pos)
        src_pos_emb.stop_gradient = True
        enc_input = src_word_emb + src_pos_emb
        return fluid.layers.dropout(
            enc_input,
            dropout_prob=self._dropout_rate,
            #seed=ModelHyperParams.dropout_seed,
            is_test=False) if self._dropout_rate else enc_input


class WrapEncoderLayer(Layer):
    """
    encoderlayer
    """

    def __init__(self, name_cope, src_vocab_size, max_length, n_layer, n_head,
                 d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, weight_sharing):
        """
        The wrapper assembles together all needed layers for the encoder.
        """
        super(WrapEncoderLayer, self).__init__(name_cope)

        self._prepare_encoder_layer = PrepareEncoderDecoderLayer(
            self.full_name(),
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout)
        self._encoder = EncoderLayer(
            self.full_name(), n_layer, n_head, d_key, d_value, d_model,
            d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd)

    def forward(self, enc_inputs):
        """forward"""
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self._prepare_encoder_layer(src_word, src_pos)
        enc_output = self._encoder(enc_input, src_slf_attn_bias)
        return enc_output