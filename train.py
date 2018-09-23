import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import json
import os
import numpy as np
import re
import nltk
import helpers
import pandas as pd
from vocab import Vocab
from  Utility import nextIteration
from LSTMRNNModel import QGLSTMRNNModel, Config

DIR='./model'
class QG(object):

    def __init__(self):
        self.vocab=Vocab()
        #self.add_variables()

    # def add_variables(self):
    #     self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, None))
    #     self.input_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))
    #     self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, None))
    #     self.answer_position = tf.placeholder(dtype=tf.float32, shape=(None, None))
    #     batch_size, _ = tf.unstack(tf.shape(self.input_placeholder))
    #     self.eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
    #     self.pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')
    #
    #     # embeddings layer
    #     self.word_embeddings_lookup = tf.get_variable('embeddings', shape=(Vocab_len, embedd_size))
    #
    #     # projection layer
    #     self.W = tf.Variable(tf.random_uniform([decoder_hidden_state_size, Vocab_len], -1, 1), dtype=tf.float32)
    #     self.b = tf.Variable(tf.zeros([Vocab_len]), dtype=tf.float32)
    #
    # def encoder(self, input):
    #     word_embeddings=self.get_embeddings(input)
    #     expanded_answer_position=tf.expand_dims(self.answer_position,2)
    #     word_embeddings_answer_position=tf.concat((word_embeddings,expanded_answer_position),2)
    #     encoder_lstm_cell = LSTMCell(num_units=encoder_hidden_state_size)
    #     ((encoder_fw_outputs, encoder_bw_outputs),
    #      (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_lstm_cell,
    #                                                                                          cell_bw=encoder_lstm_cell,
    #                                                                                          inputs=word_embeddings_answer_position,
    #                                                                                          sequence_length=self.input_length_placeholder,
    #                                                                                          dtype=tf.float32)
    #     encoder_output = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)
    #
    #     encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
    #
    #     encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    #
    #     encoder_final_state = LSTMStateTuple(
    #         c=encoder_final_state_c,
    #         h=encoder_final_state_h
    #     )
    #
    #     #decoder_lstm_cell = LSTMCell(decoder_hidden_state_size)
    #
    #     #eos_step_embedded = self.get_embeddings(self.eos_time_slice)
    #
    #     #pad_step_embedded = self.get_embeddings(self.pad_time_slice)
    #
    #     return encoder_final_state
    #
    # def get_embeddings(self, input):
    #     word_embeddings = tf.nn.embedding_lookup(params=self.word_embeddings_lookup, ids=input)
    #     return word_embeddings
    #
    # def decoder(self, encoder_final_state):
    #     decoder_lengths = self.input_length_placeholder
    #     eos_step_embedded = self.get_embeddings(self.eos_time_slice)
    #     pad_step_embedded=self.get_embeddings(self.pad_time_slice)
    #
    #     decoder_lstm_cell = LSTMCell(decoder_hidden_state_size)
    #     def loop_fn_initial():
    #         initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #         initial_input = eos_step_embedded
    #         initial_cell_state = encoder_final_state
    #         initial_cell_output = None
    #         initial_loop_state = None  # we don't need to pass any additional information
    #         return (initial_elements_finished,
    #                 initial_input,
    #                 initial_cell_state,
    #                 initial_cell_output,
    #                 initial_loop_state)
    #
    #     def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    #         def get_next_input():
    #             output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
    #             prediction = tf.argmax(output_logits, axis=1)
    #             next_input = self.get_embeddings(prediction)
    #             return next_input
    #
    #         elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    #         # defining if corresponding sequence has ended
    #
    #         finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    #         input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    #         state = previous_state
    #         output = previous_output
    #         loop_state = None
    #
    #         return (elements_finished,
    #                 input,
    #                 state,
    #                 output,
    #                 loop_state)
    #
    #     def loop_fn(time, previous_output, previous_state, previous_loop_state):
    #         if previous_state is None:  # time == 0
    #             assert previous_output is None and previous_state is None
    #             return loop_fn_initial()
    #         else:
    #             return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
    #     return tf.nn.raw_rnn(decoder_lstm_cell, loop_fn)
    #
    # def loss(self, output):
    #     decoder_outputs = output.stack()
    #     decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    #     decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    #     decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
    #     batch_size, _ = tf.unstack(tf.shape(self.input_placeholder))
    #     decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, batch_size, Vocab_len))
    #     decoder_logits = tf.transpose(decoder_logits, [1, 0, 2])
    #
    #     # decoder_prediction = tf.argmax(decoder_logits, 2)
    #
    #     on_hot_labels = tf.one_hot(indices=self.labels_placeholder, depth=Vocab_len, axis=-1)
    #     cross_entorpy = on_hot_labels * tf.log(
    #         tf.clip_by_value(decoder_logits, clip_value_min=1e-10, clip_value_max=1.0))
    #     cross_entorpy_sum = -tf.reduce_sum(cross_entorpy, 2)
    #     # the following is to mask out the padding output
    #     mask = tf.sign(tf.argmax(on_hot_labels, 2))
    #     cross_entorpy_loss = tf.reduce_sum(tf.to_float(mask) * cross_entorpy_sum)
    #     train_op = tf.train.AdamOptimizer().minimize(cross_entorpy_loss, global_step=self.global_step)
    #    return cross_entorpy_loss, train_op

    def load_data(self, data_file):
        df = pd.read_csv(data_file)
        df= df.replace(np.nan,'null',regex=True)
        # df['Answer'] = df.Answer.apply(
        #     lambda row: self.vocab.encode_list(nltk.word_tokenize(row)))
        df['Question'] = df.Question.apply(lambda row: self.vocab.encode_list(nltk.word_tokenize(row)))
        df['Answer'] = df.Answer.apply(lambda row: self.vocab.encode_list(nltk.word_tokenize(row)))
        df['Paragraph'] = df.Paragraph.apply(lambda row: self.vocab.encode_list(nltk.word_tokenize(row)))

        return df

    def run(self):

        df=self.load_data('SQUAD.csv')
        config=Config()
        model=QGLSTMRNNModel(config)
        model=model.get_model()
        with tf.Session() as sess:
            if not os.path.isdir(DIR):
                os.mkdir(DIR)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(DIR+'/checkpoint'))

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./', sess.graph)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            for i in xrange(config.iterations):
                total_loss=0
                for x_batch, sequence_lengths, y_batch, answer_position in nextIteration(df['Paragraph'].values, df['Answer'].values, df['Start position'].values, df['End position'].values, config.BATCH_SIZE):
                    # encoder_final_state=self.encoder(x_batch)
                    # decoder_outputs_ta, decoder_final_state, _=self.decoder(encoder_final_state)
                    # cross_entorpy_loss, train_op=self.loss(decoder_outputs_ta)
                    feed_dict={
                        model.input_placeholder: x_batch,
                        model.input_length_placeholder:sequence_lengths,
                        model.answer_position:answer_position,
                        model.labels_placeholder:y_batch
                    }
                    _, batch_loss=sess.run([model.train_op, model.cross_entorpy_loss],feed_dict)
                    if total_loss==0:
                        total_loss=batch_loss
                    else:
                        total_loss= total_loss + batch_loss
                #train_writer.add_summary(batch_loss, i)
                if (i + 1) % 10 == 0:
                    saver.save(sess, DIR+'/qglstmrnn', global_step=model.global_step)
                print "Loss at iteration %d : %f" %(i,total_loss)



def main():
    qg=QG()
    qg.run()

if __name__=='__main__':
    main()