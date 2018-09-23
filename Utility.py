import pandas as pd
import numpy as np

def nextIteration(features, labels, answer_start_pos, answer_end_pos, batch_size,shuffled=False):
    if shuffled:
        indices=np.random.permutation(len(labels))
        y_data=labels[indices]
        x_data=features[indices]
    else:
        y_data = labels
        x_data = features
    total_steps=int(np.ceil(len(labels)/float(batch_size)))

    for step in range(total_steps):
        start_index=step*batch_size
        y_batch=y_data[start_index:start_index+batch_size]
        x_batch=x_data[start_index:start_index+batch_size]
        start_pos_batch=answer_start_pos[start_index:start_index+batch_size]
        end_pos_batch=answer_end_pos[start_index:start_index+batch_size]

        x_batch, sequence_lengths=normalize_batch(x_batch)
        y_batch, _ =normalize_batch(y_batch,x_batch.shape[1]-2)
        answer_position=np.zeros(x_batch.shape)

        for i in range(batch_size):
            answer_position[i][start_pos_batch[i] : end_pos_batch[i]]=1
        yield x_batch, sequence_lengths, y_batch, answer_position


def normalize_batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    max_sequence_length = max_sequence_length + 1
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        inputs_batch_major[i, 0] = 1
        for j, element in enumerate(seq):
            inputs_batch_major[i, j + 1] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_lengths