# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class StackedRNNEncoder(object):
    """
    General-purpose module to encode a sequence using a stacked RNN.
    It feeds the input through a RNN and returns all the hidden states.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, num_layers, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.rnn_cells_fw = []
        self.rnn_cells_bw = []
        for layer in xrange(num_layers):
            rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
            rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
            rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
            rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)
            self.rnn_cells_fw.append(rnn_cell_fw)
            self.rnn_cells_bw.append(rnn_cell_bw)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder", reuse=tf.AUTO_REUSE):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # out Tensor shaped: [batch_size, context_len, layers_output].
            # Where layers_output are depth-concatenated forward and backward outputs.
            (out, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cells_fw, self.rnn_cells_bw, inputs, sequence_length=input_lens, dtype=tf.float32)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDirAttnFlow(object):
    """Module for bidirectional attention flow.
    [link]
    TODO: add some comments here.
    """
    def __init__(self, keep_prob, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          hidden_size: size of the hidden vectors (equal for context and question)
        """
        self.keep_prob = keep_prob
        self.w_sim1 = tf.get_variable("w_sim1", shape=(hidden_size, 1), \
            initializer=tf.contrib.layers.xavier_initializer())
        self.w_sim2 = tf.get_variable("w_sim2", shape=(hidden_size, 1), \
            initializer=tf.contrib.layers.xavier_initializer())
        self.w_sim3 = tf.get_variable("w_sim3", shape=(hidden_size, 1), \
            initializer=tf.contrib.layers.xavier_initializer())

    def build_graph(self, question_hiddens, qn_mask, context_hiddens, context_mask):
        with vs.variable_scope("BiDirAttnFlow"):
            _, context_len, hidden_sz = context_hiddens.shape
            question_len = question_hiddens.shape[1]
            print "Context_len: %d, Question_len: %d, Hidden sz: %d" % (context_len, question_len, hidden_sz)

            # 1. Calculate similary matrix (shape: (num_questions, num_contexts))
            flat_contexts = tf.reshape(context_hiddens, (-1, hidden_sz))
            flat_questions = tf.reshape(question_hiddens, (-1, hidden_sz))
            context_contribs = tf.reshape(tf.matmul(flat_contexts, self.w_sim1), (-1, context_len,)) # (batch_sz, context_len, 1)
            question_contribs = tf.reshape(tf.matmul(flat_questions, self.w_sim2), (-1, question_len,)) # (batch_sz, question_len,)

            context_hiddens_ = tf.expand_dims(context_hiddens, 1) # (batch_sz, 1, context_len, 2h)
            question_hiddens_ = tf.expand_dims(question_hiddens, 2) # (batch_sz, question_len, 1, 2h)
            # context_question_hiddens = tf.reshape(tf.multiply(context_hiddens_, question_hiddens_), (-1, hidden_sz)) # (batch_sz, question_len, context_len, 2h)
             #context_question_contribs = tf.reshape(tf.matmul(context_question_hiddens, self.w_sim3), (-1, question_len, context_len,)) # (batch_sz, question_len, context_len, )

            # w_sim3 is shape (2h, 1). context_hiddens_ is (batch_sz, 1, context_len, 2h)
            tmp1 = tf.multiply(tf.reshape(self.w_sim3, (1, 1, hidden_sz)), context_hiddens) # (batch_sz, context_len, hidden_sz)
            tmp2 = tf.matmul(tmp1, tf.transpose(question_hiddens, perm=[0,2,1]))

            # print context_question_contribs
            # print tmp2

            context_contribs_ = tf.expand_dims(context_contribs, 2) # (batch_sz, context_len, 1)
            question_contribs_ = tf.expand_dims(question_contribs, 1) # (batch_sz, 1, question_len)

            # This is the final matrix S where S_ij = w_sim^T [c_i; q_j; c_i * q_j]
            similarities = tf.add( \
                tf.add(context_contribs_, question_contribs_), \
                tmp2 \
            ) # (batch_sz, context_len, question_len)

            # print similarities

            # 2. C2Q attention.
            # Row-wise softmax of sim matrix
            # similarities_t = tf.transpose(similarities, perm=[0, 2, 1]) # shape (batch_size, context_len, question_len)
            qn_mask_ = tf.expand_dims(qn_mask, 1) # (batch_sz, 1, question_len)
            cn_mask_ = tf.expand_dims(context_mask, 2) # (batch_sz, context_len, 1)
            c2q_mask = tf.multiply(qn_mask_, cn_mask_)
            _, c2q_attn_dist = masked_softmax(similarities, c2q_mask, 1) # shape (batch_size, context_len, question_len). take softmax over values
            # Weighted sum of question hidden states
            c2q_output = tf.matmul(c2q_attn_dist, question_hiddens)

            # 3. Q2C attention.
            # Max of each row of sim matrix
            row_max_sims = tf.reduce_max(similarities, 2)
            _, q2c_attn_dist = masked_softmax(row_max_sims, context_mask, 0)
            # Weighted sum of context hidden states
            q2c_attn_dist_ = tf.expand_dims(q2c_attn_dist, 1)
            c_prime = tf.matmul(q2c_attn_dist_, context_hiddens)

            # 4. Combination vector is output
            block3 = tf.multiply(context_hiddens, c2q_output)
            block4 = tf.multiply(context_hiddens, c_prime)
            output = tf.concat([context_hiddens, c2q_output, block3, block4], 2)

            # 5. Apply dropout
            # output = tf.nn.dropout(output, self.keep_prob)

            return output

class SelfAttn(object):
    """Module for bidirectional attention flow.
    [link]
    TODO: add some comments here.
    """
    def __init__(self, keep_prob, repr_sz, hidden_sz):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          repr_size: size of the representation vectors (i.e. size of attention vectors)
          hidden_size: size of the hidden vectors (equal for context and question)
        """
        self.keep_prob = keep_prob
        self.hidden_sz = hidden_sz
        self.encoder = RNNEncoder(hidden_sz, keep_prob)
        self.w_self1 = tf.get_variable("w_self1", shape=(repr_sz, hidden_sz), \
            initializer=tf.contrib.layers.xavier_initializer())
        self.w_self2 = tf.get_variable("w_self2", shape=(repr_sz, hidden_sz), \
            initializer=tf.contrib.layers.xavier_initializer())
        self.v_self = tf.get_variable("v_self", shape=(hidden_sz, 1), \
            initializer=tf.contrib.layers.xavier_initializer())

    def build_graph(self, reprs):
        with vs.variable_scope("SelfAttn"):
            _, num_reprs, repr_sz = reprs.shape

            flat_reprs = tf.reshape(reprs, (-1, repr_sz))
            tmp1 = tf.reshape(tf.matmul(flat_reprs, self.w_self1), (-1, num_reprs, self.hidden_sz)) # (batch_sz, num_reprs, hidden_sz)
            tmp2 = tf.reshape(tf.matmul(flat_reprs, self.w_self2), (-1, num_reprs, self.hidden_sz)) # (batch_sz, num_reprs, hidden_sz)
            tmp1_ = tf.expand_dims(tmp1, 1) # (batch_sz, 1, num_reprs, hidden_sz)
            tmp2_ = tf.expand_dims(tmp2, 2) # (batch_sz, num_reprs, 1, hidden_sz)
            tmp3 = tf.tanh(tf.add(tmp1_, tmp2_)) # (batch_sz, num_reprs, num_reprs, hidden_sz) --- "i" corresponds to rows
            flat_tmp3 = tf.reshape(tmp3, (-1, self.hidden_sz))
            e = tf.reshape(tf.matmul(flat_tmp3, self.v_self), (-1, num_reprs, num_reprs, )) # (batch_sz, num_reprs, num_reprs, )

            mask = tf.ones_like(e)
            _, attn_dist = masked_softmax(e, mask, 1)

            self_outputs = tf.matmul(attn_dist, reprs)

            # concatenate self_outputs with reprs
            concat_outputs = tf.concat([reprs, self_outputs], 2) # (batch_sz, num_reprs, 2*repr_size)
            rnn_mask = tf.ones_like(concat_outputs, dtype=tf.int32)
            rnn_mask = tf.reduce_prod(rnn_mask, axis=2) # (batch_size, num_reprs, )
            output = self.encoder.build_graph(concat_outputs, rnn_mask)

            return output

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
