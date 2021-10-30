import tensorflow as tf


def kl_divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(p * (log_p - log_q), -1)
    return kl


def log_sum_exp(x):
    max_score = tf.reduce_max(x, axis=-1)
    return max_score + tf.math.log(tf.reduce_sum(tf.exp(x - tf.ones(x.shape) * max_score), axis=-1))


def top1_hard_svm(logits, labels, margin=1.):
    class_num = logits.shape[-1]
    pred_indexes = tf.argmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=class_num, dtype=tf.float32)
    one_hot_pred = tf.one_hot(pred_indexes, depth=class_num, dtype=tf.float32)

    label_score = tf.reduce_sum(one_hot_labels * logits, axis=-1)
    pred_score = tf.reduce_sum(one_hot_pred * logits, axis=-1)

    loss = pred_score + margin - label_score
    return tf.maximum(loss, 0)


def top1_smooth_svm(logits, labels, tau, margin=1.):
    class_num = logits.shape[-1]
    pred_indices = tf.argmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=class_num, dtype=tf.float32)
    one_hot_pred = tf.one_hot(pred_indices, depth=class_num, dtype=tf.float32)

    label_score = tf.reduce_sum(one_hot_labels * logits, axis=-1)
    pred_score = tf.reduce_sum(one_hot_pred * logits, axis=-1)

    loss = pred_score + margin - label_score
    # smoothing
    loss = tau * log_sum_exp(loss / tau)
    return tf.maximum(loss, 0)


def topk_smooth_svm(logits, labels, k, tau, margin=1.):
    class_num = logits.shape[-1]
    top_k_scores, _ = tf.math.top_k(logits + margin, k)
    top_k_1_scores, _ = tf.math.top_k(logits, k - 1)
    one_hot_labels = tf.one_hot(labels, depth=class_num, dtype=tf.float32)
    label_scores = tf.reduce_sum(one_hot_labels * logits, axis=1)

    top_k_with_pred = tf.reduce_sum(top_k_scores, axis=1) / (tau * k)
    top_k_with_label = label_scores + tf.reduce_sum(top_k_1_scores, axis=1) / (k * tau)

    top_k_with_pred = tau * log_sum_exp(top_k_with_pred)
    top_k_with_label = tau * log_sum_exp(top_k_with_label)

    loss = top_k_with_pred - top_k_with_label
    return tf.maximum(loss, 0)


def topk_hard_svm(logits, labels, k, margin=1.):
    class_num = logits.shape[-1]
    top_k_scores, _ = tf.math.top_k(logits + margin, k)
    top_k_1_scores, _ = tf.math.top_k(logits, k - 1)
    one_hot_labels = tf.one_hot(labels, depth=class_num, dtype=tf.float32)
    label_scores = tf.reduce_sum(one_hot_labels * logits, axis=1)

    top_k_with_pred = tf.reduce_mean(top_k_scores)
    top_k_with_label = label_scores + tf.reduce_sum(top_k_1_scores, axis=1) / k

    loss = top_k_with_pred - top_k_with_label
    return tf.maximum(loss, 0)
