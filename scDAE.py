from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import os
import sys
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.model_selection import train_test_split

# SET ENV
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth=True

# FUNCTIONS
def fc_bn(_x, _output, _phase, _scope):
	with tf.variable_scope(_scope):
		h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer())
		h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn')
		return h2

# READ RAW DATA
training_x_fileName = sys.argv[1]
training_y_fileName = sys.argv[2]
testing_x_fileName = sys.argv[3]
testing_y_fileName = sys.argv[4]

x_data = pd.read_csv(training_x_fileName, delimiter=",", dtype=np.float32).values
x_test = pd.read_csv(testing_x_fileName, delimiter=",", dtype=np.float32).values
y_data = pd.read_csv(training_y_fileName, delimiter=",", dtype=np.float32).values
y_test = pd.read_csv(testing_y_fileName, delimiter=",", dtype=np.float32).values

n_features = len(x_data[0])
n_classes = len(y_data[0])

# PLACEHOLDER
tf_raw_X = tf.placeholder(tf.float32, [None, n_features])
tf_raw_Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')
handle = tf.placeholder(tf.string, shape=[])
noise_r = tf.placeholder(tf.float32, shape=[])

# PARAMETERS
batch_size = 1303 # SET BATCH SIZE
ep_repeat = 10 # REPLACE TO PRINT FREQUENCY
prefetch_size = batch_size * 2
test_data_size = len(x_test)

learn_rate_sdae = 1e-3 # LEARNING RATE FOR DENOISING AUTOENCODER
learn_rate_sm = 1e-3 # LEARNING RATE FOR SOFTMAX
learn_rate_ft = 1e-3 # LEARNING RATE FOR FINE TUNING

keep_rate_sdae = 1
keep_rate_sm = 0.3 # DROPOUT RATE FOR SOFTMAX
keep_rate_ft = 0.3 # DROPOUT RATE FOR FINE TUNING

repeated_model_num = 1
train_sdae_eps = 3000 # EPOCHS FOR REPRESENTATION LEARNING
train_sm_eps = 3000   # EPOCHS FOR TRAINING CLASSIFIER
train_ft_eps = 1000   # EPOCHS FOR FINE-TUNING

noise_rate = 0.4 # CORRUPTION LEVEL FOR DENOSING AUTOENCODER

max_accr= 0.0

# DATASET & ITERATOR
dataset_train = tf.data.Dataset.from_tensor_slices((tf_raw_X, tf_raw_Y))
dataset_train = dataset_train.shuffle(buffer_size=batch_size * 2)
dataset_train = dataset_train.repeat(ep_repeat).batch(batch_size).prefetch(prefetch_size)
iterator_train = dataset_train.make_initializable_iterator()

dataset_test = tf.data.Dataset.from_tensor_slices((tf_raw_X, tf_raw_Y))
dataset_test = dataset_test.batch(test_data_size)
iterator_test = dataset_test.make_initializable_iterator()

iter = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)

tf_X, tf_Y = iter.get_next()

# MODEL STRUCTURE
n_in = n_features
n_en_h1 = 1000 # NUMBER OF HIDDEN NODES IN FIRST LAYER OF ENCODER
n_en_h2 = 500  # NUMBER OF HIDDEN NODES IN SECOND LAYER OF ENCODER
n_code_h = 125 # NUMBER OF HIDDEN NODES FOR LATENT REPRESENTATIONS
n_de_h1 = 500  # NUMBER OF HIDDEN NODES IN FIRST LAYER OF DECODER
n_de_h2 = 1000 # NUMBER OF HIDDEN NODES IN SECOND LAYER OF DECODER
n_out = n_features

n_sm_h1 = n_code_h
n_sm_h2 = n_code_h
n_sm_out = n_classes

print("# feature:", n_features, "# classes:", n_classes, "# train sample:", len(x_data))

# MODEL FUNCTIONS
def encoder(_X, _keep_prob, _phase, _noise_rate):
	en1 = tf.nn.dropout(tf.nn.elu(fc_bn(tf.contrib.keras.layers.GaussianNoise(stddev=_noise_rate)(_X), n_en_h1, _phase, "en1")), _keep_prob)
	en2 = tf.nn.dropout(tf.nn.tanh(fc_bn(en1, n_en_h2, _phase, "en2")), _keep_prob)
	code = fc_bn(en2, n_code_h, _phase, "code")
	return code

def decoder(_code, _keep_prob, _phase):
	de2 = tf.nn.dropout(tf.nn.tanh(fc_bn(_code, n_de_h1, _phase, "de2")), _keep_prob)
	de3 = tf.nn.dropout(tf.nn.elu(fc_bn(de2, n_de_h2, _phase, "de3")), _keep_prob)
	decode = tf.nn.sigmoid(fc_bn(de3, n_out, _phase, "decode"))
	return decode

def softmax(_code,_keep_prob, _phase):
	fc1 = tf.nn.dropout(tf.nn.elu(fc_bn(_code, n_sm_h1, _phase, "fc1")), _keep_prob)
	fc2 = tf.nn.dropout(tf.nn.elu(fc_bn(fc1, n_sm_h2, _phase, "fc2")), _keep_prob)
	sm = tf.nn.softmax(fc_bn(fc2, n_sm_out, _phase, "sm"))
	return sm

# MODEL
code = encoder(tf_X, keep_prob, phase, noise_r)
decode = decoder(code, keep_prob, phase)
sm_out = softmax(code, keep_prob, phase)

print ("MODEL READY") 

# DEFINE LOSS AND OPTIMIZER
sdae_cost = tf.reduce_mean((decode - tf_X) ** 2)
sm_cost = tf.reduce_mean(-tf.reduce_sum(tf_Y * tf.log(sm_out + 1e-10), axis = 1))
ft_cost = sdae_cost + sm_cost

train_op_sdae = tf.train.RMSPropOptimizer(learning_rate=learn_rate_sdae).minimize(sdae_cost)
train_op_sm = tf.train.AdamOptimizer(learning_rate=learn_rate_sm).minimize(sm_cost)
train_op_ft = tf.train.AdamOptimizer(learning_rate=learn_rate_ft).minimize(ft_cost)

# ACCURACY
pred = tf.argmax(sm_out, 1)
label = tf.argmax(tf_Y, 1)
correct_pred = tf.equal(pred, label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
_accuracy = tf.Variable(0)

print ("FUNCTIONS READY") 

# START SESSION
sess = tf.Session(config=config)
handle_train = sess.run(iterator_train.string_handle())
handle_test = sess.run(iterator_test.string_handle())
saver = tf.train.Saver()

print ("START OPTIMIZATION & TESTING")
for model_num in xrange(repeated_model_num):
	sess.run(tf.global_variables_initializer())	

	# SET OPS & FEED_DICT
	sdae_ops = [sdae_cost, train_op_sdae, _accuracy]
	sm_ops = [sm_cost, train_op_sm, accuracy]
	ft_ops = [ft_cost, train_op_ft, accuracy]

	sdae_feed_dict = {handle: handle_train, keep_prob : keep_rate_sdae, phase: True, noise_r: noise_rate}
	sm_feed_dict = {handle: handle_train, keep_prob: keep_rate_sm, phase: True, noise_r: noise_rate}
	ft_feed_dict = {handle: handle_train, keep_prob: keep_rate_ft, phase: True, noise_r: noise_rate}

	# SDAE, SM, FT
	for temp_ep, meta_step, temp_ops, temp_feed_dict in zip([train_sdae_eps, train_sm_eps, train_ft_eps], ["_sdae", "_sm", "_ft"], [sdae_ops, sm_ops, ft_ops], [sdae_feed_dict, sm_feed_dict, ft_feed_dict]):	

		for ep in xrange(temp_ep/ep_repeat):
			sess.run(iterator_train.initializer, feed_dict={tf_raw_X: x_data, tf_raw_Y: y_data})

			# REPEAT NUMBER OF EPS WITHOUT BREAK. SET BY ep_repeat
			while True: 
				try:
					cur_cost_val, _, cur_accuracy = sess.run(temp_ops, feed_dict = temp_feed_dict)
				except tf.errors.OutOfRangeError:
					break

			# EXECUTED PER ep_repeat
			print("Model#:%02d," % model_num, "Ep:%04d," % (ep*ep_repeat), "Cost" + meta_step + ":%.9f" % cur_cost_val, end='')

			if meta_step != "_sdae":
				sess.run(iterator_test.initializer, feed_dict={tf_raw_X: x_test, tf_raw_Y: y_test})
				cur_acc, cur_pred, cur_label = sess.run([accuracy, pred, label], feed_dict = {handle: handle_test, keep_prob: 1.0, phase: False, noise_r: 0})
				print(", Test_Accr:%.9f," % cur_acc, "Train_batch_accr:%.9f" % cur_accuracy, end='')

				# STORE MAX MODEL
				if max_accr < float(cur_acc):
					max_accr = cur_acc
					max_pred = cur_pred
					max_label = cur_label

			print("")

		print(meta_step + "_part is DONE!")

	print("\nACCURACY OF THIS MODEL : %.6f" % max_accr)

	np.savetxt("./Classified_result_with_accuracy_of" + "_" + str(max_accr) + "_" + str(n_features) + "_features", max_pred, fmt="%.0f", delimiter=",")
	max_accr= 0.0


