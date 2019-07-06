# lstm + crf
import tensorflow as tf 

from config import cfg

class Model(object):
	def __init__(self,corpus):
		self.vocab_size = corpus.vocab_size
		self.label_size = corpus.label_size
		self.embed_size = cfg.embed_size
		self.hidden_size = cfg.hidden_size
		self.num_layers = cfg.num_layers
		self.batch_size = cfg.batch_size

	# placeholder
		self.x = tf.placeholder(tf.int32,[None,None]) #[batch,length]
		self.x_lengths = tf.placeholder(tf.int32,[None]) # [batch]
		self.y = tf.placeholder(tf.int32,[None,None])
		# self.batch_size = tf.placeholder(tf.int32,[None])
	# Linear layer
		weights = {
			"in":tf.Variable(tf.random_normal([self.embed_size,self.hidden_size]),dtype = tf.float32,name = 'weights_in'),
			"out":tf.Variable(tf.random_normal([self.hidden_size*2,self.label_size]),dtype = tf.float32,name = 'weights_out')
		}
		biases = {
			"in":tf.Variable(tf.constant(0.1,shape = [self.hidden_size]),dtype = tf.float32,name = 'biases_in'),
			"out":tf.Variable(tf.constant(0.1,shape = [self.label_size]),dtype = tf.float32,name = 'biases_out')
		}
	# embeddings layer
		self.embedding = tf.get_variable('embedding',[self.vocab_size,self.embed_size],
										initializer = tf.initializers.variance_scaling)
		x_embed = tf.nn.embedding_lookup(self.embedding,self.x) #[batch,length,embed_size]
		x_embed = tf.reshape(x_embed,[-1,self.embed_size]) #[batch*length,embed_size]
	# linear layer
		x_in = tf.matmul(x_embed,weights['in']) + biases['in'] #[batch*length,hidden_size]
		x_in = tf.reshape(x_in,[self.batch_size,-1,self.hidden_size])
	# bi-LSTM layer
		fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,name = 'lstm',forget_bias = 0.0,
											state_is_tuple = True)
		bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,name = 'lstm',forget_bias = 0.0,
											state_is_tuple = True)

		mfw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * self.num_layers,state_is_tuple = True)
		mbw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * self.num_layers,state_is_tuple = True)
		fw_init_state = mfw_cell.zero_state(self.batch_size,dtype = tf.float32)
		bw_init_state = mbw_cell.zero_state(self.batch_size,dtype = tf.float32)

		output,hidden = tf.nn.bidirectional_dynamic_rnn(mfw_cell,mbw_cell,x_in,initial_state_fw = fw_init_state,
                        initial_state_bw = bw_init_state,dtype = tf.float32)
		# output: [num_directions,batch,length,hidden_size]
		# hidden = (h,c) : [num_layers * num_directions,batch,hidden_size]
	# linear layer
		fw_out,bw_out = output
		outputs = tf.concat([fw_out,bw_out],axis = 2) # [batch,length,hidden_size * 2]
		outputs = tf.reshape(outputs,[-1,self.hidden_size * 2])
		results = tf.matmul(outputs,weights['out']) + biases['out']
		logits = tf.reshape(results,[self.batch_size,-1,self.label_size])
		self.logits = logits
		# [batch,length,label_size]
	# crf layer
		log_likelihood,transition_params = tf.contrib.crf.crf_log_likelihood(logits,self.y,self.x_lengths)
		# logits: [batch,length,label_size] 
		# y: [batch,length] true label
		# x_lengths: [batch] the length of x batch, to mask the padding tokens
		# log_likelihood: [batch] the log-likelihood loss of the x
		# transition_params: [label_size,label_size], the transition matrix of crf
		self.transition_params = transition_params
	# caculate loss,train mode
		self.loss = tf.reduce_mean(-log_likelihood)
	# update patameters
		self.train_op = tf.train.AdamOptimizer(cfg.lr).minimize(self.loss)

	def forward(self,sess,words,word_lengths = None,labels = None,mode = 'train'):
		feed_dict = {self.x: words,
					 self.x_lengths: word_lengths,
					 self.y: labels,}
		if mode == 'train': # train mode, update parameters
			feed_out = [self.train_op,self.loss]
		elif mode == 'valid': # valid mode, not update parameters
			feed_out = [self.loss,self.logits,self.transition_params]
		elif mode == 'infer': # infer/test mode, given words to predict 
			feed_dict = {self.x: words}
			feed_out = [self.logits,self.transition_params]
		out = sess.run(feed_out,feed_dict = feed_dict)
		return out

