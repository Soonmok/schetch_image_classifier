import tensorflow as tf


def evaluate_top_k_score(labels, y_pred, k=5):
    scores = tf.math.in_top_k(labels, y_pred, k)
    return scores


def evaluate_map(labels, y_pred, k=5):
    _, indices = tf.math.top_k(y_pred, k)
    to_top_k = tf.ones((1, k), dtype=tf.int32)
    top_k_labels = tf.matmul(tf.expand_dims(tf.cast(labels, dtype=tf.int32), axis=-1), to_top_k)
    mask = tf.cast(tf.equal(top_k_labels, indices), dtype=tf.float32)
    weighted_matrix = tf.expand_dims(tf.range(1, 0, -1 / k, dtype=tf.float32), axis=-1)
    scores = tf.matmul(mask, weighted_matrix)
    return scores


def evaluate_confident_top1(y_pred):
    p = tf.nn.softmax(y_pred)
    pred_score = tf.maximum(p, axis=-1)
    return pred_score


def evaluate_confident_topk(y_pred, k=5):
    p = tf.nn.softmax(y_pred)
    top_k_prob, indices = tf.math.top_k(p, k)
    pred_score = tf.reduce_sum(top_k_prob, axis=-1) / k
    return pred_score