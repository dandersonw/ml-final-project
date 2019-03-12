import tensorflow as tf
import numpy as np

from .script import SCRIPTS
from .train import deintern_decode_results


# TODO: length penalty? other assorted heuristics?
def beam_search_decode(*,
                       encoder_output,
                       encoder_state,
                       decoder,
                       from_script,
                       to_script,
                       max_len=20,
                       num_beams=5,
                       k_best=1):
    assert num_beams >= k_best

    start_token = SCRIPTS[to_script].intern_char('<start>')
    end_token = SCRIPTS[to_script].intern_char('<end>')
    batch_size = int(encoder_output.shape[0])
    vocab_size = SCRIPTS[to_script].vocab_size

    hyp = np.zeros([batch_size, num_beams, max_len], dtype=np.int64)
    hyp_prob = np.full([batch_size, num_beams], np.inf)  # neg log prob
    done = np.zeros([batch_size, num_beams], dtype=np.bool)

    decoder_state = decoder.make_initial_state(encoder_state)
    # TODO: consider what to do if decoder state is more complicated
    hyp_state = np.zeros([batch_size, num_beams, decoder_state.shape[-1]],
                         dtype=np.float32)

    hyp_state[:, 0] = decoder_state
    hyp[:, 0, 0] = start_token
    hyp_prob[:, 0] = 0

    must_be_pad = np.full([vocab_size], np.inf)
    must_be_pad[0] = 0

    t = 1
    while t < max_len and not np.all(done):
        flat_in = tf.reshape(hyp[:, :, t - 1],
                             [batch_size * num_beams])
        flat_state = tf.reshape(hyp_state,
                                [batch_size * num_beams, -1])

        pred, state_pred = decoder(flat_in,
                                   flat_state,
                                   encoder_output)
        pred = -1 * np.log(tf.nn.softmax(pred))
        pred = np.reshape(pred, [batch_size, num_beams, vocab_size])
        pred[done] = must_be_pad
        prob = pred + np.expand_dims(hyp_prob, 2)
        flattened_prob = np.reshape(prob, [batch_size, -1])

        branching_factor = min(num_beams, vocab_size)
        flat_idx = np.argsort(flattened_prob)[:, :branching_factor]
        beam_choice, token_choice = np.unravel_index(np.ravel(flat_idx),
                                                     [num_beams, vocab_size])
        beam_choice = np.reshape(beam_choice, [batch_size, num_beams])
        token_choice = np.reshape(token_choice, [batch_size, num_beams])

        # I guess ideally this big block below would be done in numpy code,
        # but my kungfu isn't strong enough
        done_ = np.zeros_like(done)
        hyp_ = np.zeros_like(hyp)
        hyp_state_ = np.zeros_like(hyp_state)
        hyp_prob_ = np.zeros_like(hyp_prob)
        for i in range(batch_size):
            for j in range(num_beams):
                parent_beam = beam_choice[i, j]
                chosen_token = token_choice[i, j]
                done_[i, j] = np.logical_or(done[i, parent_beam],
                                            chosen_token == end_token)
                hyp_[i, j] = hyp[i, parent_beam]
                hyp_[i, j, t] = chosen_token
                hyp_state_[i, j] = state_pred[i * num_beams + parent_beam]
                hyp_prob_[i, j] = prob[i, parent_beam, chosen_token]
        done = done_
        hyp = hyp_
        hyp_state = hyp_state_
        hyp_prob = hyp_prob_
        # maybe I could rotate the arrays or something?
        # I guess performance isn't a big issue
        t += 1

    hyp = hyp[:, :, 1:]  # chop off the <start> token we put
    return hyp[:, 0]
