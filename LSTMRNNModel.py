import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

class Config (object):
    embedd_size = 265
    output_size = 256
    iterations=100
    BATCH_SIZE = 10
    encoder_hidden_state_size = 256
    decoder_hidden_state_size = encoder_hidden_state_size * 2
    l2 = 0.02
    lr = 0.01
    Vocab_len = 95335
    MAX_SEQUENCE = 50
    EOS = 0
    PAD = 1

class QGLSTMRNNModel(object):

    def __init__ (self, config):
        self.config=config

    def _add_variables(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.input_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.answer_position = tf.placeholder(dtype=tf.float32, shape=(None, None))
        batch_size, _ = tf.unstack(tf.shape(self.input_placeholder))
        self.eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        self.pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        # embeddings layer
        self.word_embeddings_lookup = tf.get_variable('embeddings', shape=(self.config.Vocab_len, self.config.embedd_size))

        # projection layer
        self.W = tf.Variable(tf.random_uniform([self.config.decoder_hidden_state_size, self.config.Vocab_len], -1, 1), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.config.Vocab_len]), dtype=tf.float32)

    def _encoder(self):
        word_embeddings=self._get_embeddings(self.input_placeholder)
        expanded_answer_position=tf.expand_dims(self.answer_position,2)
        word_embeddings_answer_position=tf.concat((word_embeddings,expanded_answer_position),2)
        encoder_lstm_cell = LSTMCell(num_units=self.config.encoder_hidden_state_size)
        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_lstm_cell,
                                                                                             cell_bw=encoder_lstm_cell,
                                                                                             inputs=word_embeddings_answer_position,
                                                                                             sequence_length=self.input_length_placeholder,
                                                                                             dtype=tf.float32)
        encoder_output = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)

        encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        #decoder_lstm_cell = LSTMCell(decoder_hidden_state_size)

        #eos_step_embedded = self.get_embeddings(self.eos_time_slice)

        #pad_step_embedded = self.get_embeddings(self.pad_time_slice)

        return encoder_final_state

    def _get_embeddings(self, input):
        word_embeddings = tf.nn.embedding_lookup(params=self.word_embeddings_lookup, ids=input)
        return word_embeddings

    def _decoder(self, encoder_final_state):
        decoder_lengths = self.input_length_placeholder
        eos_step_embedded = self._get_embeddings(self.eos_time_slice)
        pad_step_embedded=self._get_embeddings(self.pad_time_slice)

        decoder_lstm_cell = LSTMCell(self.config.decoder_hidden_state_size)
        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = eos_step_embedded
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = self._get_embeddings(prediction)
                return next_input

            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            # defining if corresponding sequence has ended

            finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
        return tf.nn.raw_rnn(decoder_lstm_cell, loop_fn)

    def _loss(self, decoder_output):
        decoder_outputs = decoder_output.stack()
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
        batch_size, _ = tf.unstack(tf.shape(self.input_placeholder))
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, batch_size, self.config.Vocab_len))
        decoder_logits = tf.transpose(decoder_logits, [1, 0, 2])

        # decoder_prediction = tf.argmax(decoder_logits, 2)

        on_hot_labels = tf.one_hot(indices=self.labels_placeholder, depth=self.config.Vocab_len, axis=-1)
        cross_entorpy = on_hot_labels * tf.log(
            tf.clip_by_value(decoder_logits, clip_value_min=1e-10, clip_value_max=1.0))
        cross_entorpy_sum = -tf.reduce_sum(cross_entorpy, 2)
        # the following is to mask out the padding output
        mask = tf.sign(tf.argmax(on_hot_labels, 2))
        self.cross_entorpy_loss = tf.reduce_sum(tf.to_float(mask) * cross_entorpy_sum)
        tf.summary.scalar('cross_entropy', self.cross_entorpy_loss)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cross_entorpy_loss, global_step=self.global_step)

    def get_model(self):
        self._add_variables()
        encoder_final_state=self._encoder()
        decoder_outputs_ta, decoder_final_state, _ =self._decoder(encoder_final_state)
        self._loss(decoder_outputs_ta)
        return self




