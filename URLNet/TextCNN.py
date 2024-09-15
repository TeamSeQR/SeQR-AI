import tensorflow as tf 
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Lambda    # 새로운 import 추가
from tensorflow.keras.initializers import GlorotUniform  # Xavier Initializer를 대체하기 위한 Glorot Uniform

class TextCNN(object): 
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, \
        word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0, \
        filter_sizes=[3,4,5,6], mode=0): 
        
        # Keras Input 정의: 입력 데이터를 받을 Placeholder 대신 사용
        if mode == 4 or mode == 5: 
            # 'char_ngram' 형태의 데이터를 위한 입력 정의
            self.input_x_char = Input(shape=(None, None), dtype='int32', name="input_x_char")
            # 'char_ngram' 데이터의 패딩 인덱스를 위한 입력 정의
            self.input_x_char_pad_idx = Input(shape=(None, None, embedding_size), dtype='float32', name="input_x_char_pad_idx")
        
        if mode == 4 or mode == 5 or mode == 2 or mode == 3: 
            self.input_x_word = Input(shape=(None,), dtype='int32', name="input_x_word")

        if mode == 1 or mode == 3 or mode == 5: 
            self.input_x_char_seq = Input(shape=(None,), dtype='int32', name="input_x_char_seq")

        # 출력(label)과 dropout 확률에 대한 입력 정의
        self.input_y = Input(shape=(None,), dtype='float32', name="input_y")
        self.dropout_keep_prob = Input(shape=(), dtype='float32', name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)  # 정규화 항을 위한 L2 손실 초기값

        # 임베딩 레이어 정의
        with tf.name_scope("embedding"): 
            # 첫 번째 블록: 'char_ngram' 형태 데이터의 임베딩 처리
            if mode == 4 or mode == 5: 
                # Embedding 레이어 사용하여 'char_ngram' 데이터의 임베딩 벡터 생성
                self.embedded_x_char = Embedding(char_ngram_vocab_size, embedding_size, name="char_emb_w")(self.input_x_char)
                # 패딩 인덱스와 곱하여 임베딩 벡터 정규화
                self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)

            # 두 번째 블록: 'word' 형태 데이터의 임베딩 처리
            if mode == 2 or mode == 3 or mode == 4 or mode == 5: 
                # Embedding 레이어 사용하여 'word' 데이터의 임베딩 벡터 생성
                self.embedded_x_word = Embedding(word_ngram_vocab_size, embedding_size, name="word_emb_w")(self.input_x_word)

            # 세 번째 블록: 'char_seq' 형태 데이터의 임베딩 처리
            if mode == 1 or mode == 3 or mode == 5: 
                # Embedding 레이어 사용하여 'char_seq' 데이터의 임베딩 벡터 생성
                self.embedded_x_char_seq = Embedding(char_vocab_size, embedding_size, name="char_seq_emb_w")(self.input_x_char_seq)
                
                # 임베딩 벡터를 3차원으로 확장하여 Conv 레이어 입력 형태에 맞춤
                ## self.char_x_expanded = tf.expand_dims(self.embedded_x_char_seq, -1)
                # Lambda 레이어로 tf.expand_dims 대체
                self.char_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.embedded_x_char_seq)  # 수정된 부분


            # 임베딩 벡터 후처리: 필요한 경우에만 수행
            if mode == 4 or mode == 5: 
                # 'char_ngram' 임베딩 벡터의 합계 계산
                self.sum_ngram_x_char = tf.reduce_sum(self.embedded_x_char, 2)         
                # 합계 계산 후 'word' 임베딩 벡터와 결합
                self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word) 
                # 결합된 벡터를 4차원으로 확장
                ## self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
                 # Lambda 레이어로 tf.expand_dims 대체
                self.sum_ngram_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.sum_ngram_x)  # 수정된 부분


            if mode == 2 or mode == 3: 
                # 'word' 임베딩 벡터를 4차원으로 확장
                ##self.sum_ngram_x_expanded = tf.expand_dims(self.embedded_x_word, -1)
                # Lambda 레이어로 tf.expand_dims 대체
                self.sum_ngram_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.embedded_x_word)  # 수정된 부분

        ########################### WORD CONVOLUTION LAYER ################################
        if mode == 2 or mode == 3 or mode == 4 or mode == 5: 
            pooled_x = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool_%s" % filter_size): 
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b") 
                    #w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    w = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="w") 
                    conv = tf.nn.conv2d(
                        self.sum_ngram_x_expanded,
                        w,
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu") 
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, word_seq_len - filter_size + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool") 
                    pooled_x.append(pooled) 
        
            num_filters_total = 256 * len(filter_sizes) 
            self.h_pool = tf.concat(pooled_x, 3)
            self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")  
            self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x") 

        ########################### CHAR CONVOLUTION LAYER ###########################
        if mode == 1 or mode == 3 or mode == 5: 
            pooled_char_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                    #w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    w = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.char_x_expanded,
                        w,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, char_seq_len - filter_size + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool")
                    pooled_char_x.append(pooled) 
            num_filters_total = 256*len(filter_sizes) 
            self.h_char_pool = tf.concat(pooled_char_x, 3)
            self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total], name="pooled_char_x")
            self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob, name="dropout_char_x")
        
        ############################### CONCAT WORD AND CHAR BRANCH ############################
        if mode == 3 or mode == 5: 
            with tf.name_scope("word_char_concat"): 
                #ww = tf.get_variable("ww", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
                bw = tf.Variable(tf.constant(0.1, shape=[512]), name="bw") 
                ww = tf.Variable(GlorotUniform()(shape=(num_filters_total, 512)), name="ww")
                l2_loss += tf.nn.l2_loss(ww) 
                l2_loss += tf.nn.l2_loss(bw) 
                #word_output = tf.nn.xw_plus_b(self.h_drop, ww, bw)
                word_output = tf.linalg.matmul(self.h_drop, ww) + bw

                #wc = tf.get_variable("wc", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
                wc = tf.Variable(GlorotUniform()(shape=(num_filters_total, 512)), name="wc")
                bc = tf.Variable(tf.constant(0.1, shape=[512]), name="bc") 
                l2_loss += tf.nn.l2_loss(wc)
                l2_loss += tf.nn.l2_loss(bc)
                #char_output = tf.nn.xw_plus_b(self.char_h_drop, wc, bc) 
                char_output = tf.linalg.matmul(self.char_h_drop, wc) + bc
                
            
                self.conv_output = tf.concat([word_output, char_output], 1)              
        elif mode == 2 or mode == 4: 
            self.conv_output = self.h_drop 
        elif mode == 1: 
            self.conv_output = self.char_h_drop        

        ################################ RELU AND FC ###################################
        with tf.name_scope("output"): 
            #w0 = tf.get_variable("w0", shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer())
            w0 = tf.Variable(GlorotUniform()(shape=[1024, 512]), name="w0")
            b0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b0") 
            l2_loss += tf.nn.l2_loss(w0) 
            l2_loss += tf.nn.l2_loss(b0) 
            output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)
            
            #w1 = tf.get_variable("w1", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer()) 
            w1 = tf.Variable(GlorotUniform()(shape=[512, 256]), name="w1")
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1") 
            l2_loss += tf.nn.l2_loss(w1) 
            l2_loss += tf.nn.l2_loss(b1) 
            output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)
            
            #w2 = tf.get_variable("w2", shape=[256,128], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.Variable(GlorotUniform()(shape=[256, 128]), name="w2")
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2") 
            l2_loss += tf.nn.l2_loss(w2) 
            l2_loss += tf.nn.l2_loss(b2) 
            output2 = tf.nn.relu(tf.matmul(output1, w2) + b2) 
            
            #w = tf.get_variable("w", shape=(128, 2), initializer=tf.contrib.layers.xavier_initializer()) 
            w = tf.Variable(GlorotUniform()(shape=(128, 2)), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b") 
            l2_loss += tf.nn.l2_loss(w) 
            l2_loss += tf.nn.l2_loss(b) 
            
            #self.scores = tf.nn.xw_plus_b(output2, w, b, name="scores") 
            self.scores = tf.linalg.matmul(output2, w) + b
            self.predictions = tf.argmax(self.scores, 1, name="predictions") 

        with tf.name_scope("loss"): 
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y) 
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"): 
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1)) 
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy") 
