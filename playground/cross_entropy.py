import tensorflow as tf
tf.InteractiveSession()

#####################################################################################
ground_truth_class_idx = 2  # Switch between: 0, 1 or 2
class_probabilities = [0.2, 0.3, 0.5]  # dummy probabilities predicted for each class
#####################################################################################

ground_truth = tf.constant([ground_truth_class_idx], dtype=tf.int32)
prediction_sparse = tf.constant([class_probabilities], dtype=tf.float32)
assert tf.rank(ground_truth).eval() == (tf.rank(prediction_sparse).eval() - 1)

cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth,
                                                                    logits=prediction_sparse)

print "Cross entropy for class:", ground_truth_class_idx, \
      "(confidence: " + str(class_probabilities[ground_truth_class_idx]) + ") is:", cross_entropy_loss.eval()[0]

