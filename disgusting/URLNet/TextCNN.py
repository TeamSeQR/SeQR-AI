import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Lambda, Multiply, Add, Conv2D, ReLU, MaxPooling2D, Concatenate, Reshape, Dropout
from tensorflow.keras.initializers import GlorotUniform

class TextCNN(object): 
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, \
        word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0, \
        filter_sizes=[3,4,5,6], mode=0): 
        
        # Keras Input 정의
        if mode == 4 or mode == 5: 
            self.input_x_char = Input(shape=(None, None), dtype='int32', name="input_x_char")
            self.input_x_char_pad_idx = Input(shape=(None, None, embedding_size), dtype='float32', name="input_x_char_pad_idx")
        
        if mode == 4 or mode == 5 or mode == 2 or mode == 3: 
            self.input_x_word = Input(shape=(None,), dtype='int32', name="input_x_word")

        if mode == 1 or mode == 3 or mode == 5: 
            self.input_x_char_seq = Input(shape=(None,), dtype='int32', name="input_x_char_seq")

        # 출력과 dropout 확률에 대한 입력 정의
        self.input_y = Input(shape=(None,), dtype='float32', name="input_y")
        self.dropout_keep_prob = Input(shape=(), dtype='float32', name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)  # 정규화 항을 위한 L2 손실 초기값

        # 임베딩 레이어 정의
        with tf.name_scope("embedding"): 
            if mode == 4 or mode == 5: 
                self.embedded_x_char = Embedding(char_ngram_vocab_size, embedding_size, name="char_emb_w")(self.input_x_char)
                # tf.multiply 대신 keras.layers.Multiply 사용
                self.embedded_x_char = Multiply()([self.embedded_x_char, self.input_x_char_pad_idx])

            if mode == 2 or mode == 3 or mode == 4 or mode == 5: 
                self.embedded_x_word = Embedding(word_ngram_vocab_size, embedding_size, name="word_emb_w")(self.input_x_word)

            if mode == 1 or mode == 3 or mode == 5: 
                self.embedded_x_char_seq = Embedding(char_vocab_size, embedding_size, name="char_seq_emb_w")(self.input_x_char_seq)
                # Lambda 레이어로 tf.expand_dims 대체
                self.char_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.embedded_x_char_seq)

            if mode == 4 or mode == 5: 
                # tf.reduce_sum 대신 keras.layers.Lambda 사용
                self.sum_ngram_x_char = Lambda(lambda x: tf.reduce_sum(x, axis=2))(self.embedded_x_char)
                # tf.add 대신 keras.layers.Add 사용
                self.sum_ngram_x = Add()([self.sum_ngram_x_char, self.embedded_x_word])
                # Lambda 레이어로 tf.expand_dims 대체
                self.sum_ngram_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.sum_ngram_x)

            if mode == 2 or mode == 3: 
                self.sum_ngram_x_expanded = Lambda(lambda x: tf.expand_dims(x, -1))(self.embedded_x_word)

        ########################### WORD CONVOLUTION LAYER ################################
        if mode == 2 or mode == 3 or mode == 4 or mode == 5: 
            pooled_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool_%s" % filter_size):
                    # Conv2D 레이어 사용
                    conv = Conv2D(
                        256,  # 필터 개수
                        (filter_size, embedding_size),  # 커널 크기
                        strides=(1, 1),
                        padding='valid',
                        name=f"conv_{filter_size}"
                    )(self.sum_ngram_x_expanded)
                    
                    # tf.nn.relu 대신 keras.layers.ReLU 사용
                    h = ReLU(name="relu")(conv)
                    
                    # MaxPooling2D 레이어 사용
                    pooled = MaxPooling2D(
                        pool_size=(word_seq_len - filter_size + 1, 1),
                        strides=(1, 1),
                        padding='valid',
                        name="pool"
                    )(h)
                    pooled_x.append(pooled)

            num_filters_total = 256 * len(filter_sizes) 
            #self.h_pool = tf.concat(pooled_x, 3)
            self.h_pool = Concatenate(axis=3)(pooled_x)  # axis=3은 마지막 차원에서의 결합을 의미
            #self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")  
            #self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x") 
            # Keras의 Reshape 레이어 사용
            self.x_flat = Reshape((-1, num_filters_total), name="pooled_x")(self.h_pool) 
            # Keras의 Dropout 레이어 사용
            ## self.h_drop = Dropout(1 - self.dropout_keep_prob, name="dropout_x")(self.x_flat)
            self.h_drop = Dropout(rate=0.5, name="dropout_x")(self.x_flat)  # rate=0.5 고정 비율 사용


        ########################### CHAR CONVOLUTION LAYER ###########################
        if mode == 1 or mode == 3 or mode == 5: 
            pooled_char_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                    # Conv2D 레이어 사용
                    conv = Conv2D(
                        256,
                        (filter_size, embedding_size),
                        strides=(1, 1),
                        padding='valid',
                        name=f"char_conv_{filter_size}"
                    )(self.char_x_expanded)
                    
                    # tf.nn.relu 대신 keras.layers.ReLU 사용
                    h = ReLU(name="relu")(conv)

                    # MaxPooling2D 레이어 사용
                    pooled = MaxPooling2D(
                        pool_size=(char_seq_len - filter_size + 1, 1),
                        strides=(1, 1),
                        padding='valid',
                        name="char_pool"
                    )(h)
                    pooled_char_x.append(pooled)
            
            num_filters_total = 256 * len(filter_sizes) 
            #self.h_char_pool = tf.concat(pooled_char_x, 3)
            #self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total], name="pooled_char_x")
            #self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob, name="pooled_char_x")
            # Keras의 Reshape 레이어 사용
            self.h_char_pool = Concatenate(axis=3)(pooled_char_x)  # axis=3은 마지막 차원에서의 결합을 의미
            self.char_x_flat = Reshape((-1, num_filters_total), name="pooled_char_x")(self.h_char_pool) 
            # Keras의 Dropout 레이어 사용
            ## self.char_h_drop = Dropout(1 - self.dropout_keep_prob, name="pooled_char_x")(self.char_x_flat)
            self.char_h_drop = Dropout(rate=0.5, name="pooled_char_x")(self.char_x_flat)  # 고정된 드롭아웃 비율 사용


        # 이후 코드는 동일하게 유지
        ############################### CONCAT WORD AND CHAR BRANCH ############################
        # 나머지 모델 구성 코드는 그대로 유지
        if mode == 3 or mode == 5: 
            with tf.name_scope("word_char_concat"): 
                #ww = tf.get_variable("ww", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
                bw = tf.Variable(tf.constant(0.1, shape=[512]), name="bw") 
                ww = tf.Variable(GlorotUniform()(shape=(num_filters_total, 512)), name="ww")
                l2_loss += tf.nn.l2_loss(ww) 
                l2_loss += tf.nn.l2_loss(bw) 
                #word_output = tf.nn.xw_plus_b(self.h_drop, ww, bw)
                #word_output = tf.linalg.matmul(self.h_drop, ww) + bw
                word_output = Dense(512, activation=None, name="word_output")(self.h_drop)  # ww와 bw를 합친 Dense 레이어

                #wc = tf.get_variable("wc", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
                wc = tf.Variable(GlorotUniform()(shape=(num_filters_total, 512)), name="wc")
                bc = tf.Variable(tf.constant(0.1, shape=[512]), name="bc") 
                l2_loss += tf.nn.l2_loss(wc)
                l2_loss += tf.nn.l2_loss(bc)
                #char_output = tf.nn.xw_plus_b(self.char_h_drop, wc, bc) 
                #char_output = tf.linalg.matmul(self.char_h_drop, wc) + bc
                char_output = Dense(512, activation=None, name="char_output")(self.char_h_drop)  # wc와 bc를 합친 Dense 레이어
                
            
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
            #output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)
            output0 = Dense(512, activation='relu', name="output0")(self.conv_output)  # w0와 b0를 합친 Dense 레이어
            
            #w1 = tf.get_variable("w1", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer()) 
            w1 = tf.Variable(GlorotUniform()(shape=[512, 256]), name="w1")
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1") 
            l2_loss += tf.nn.l2_loss(w1) 
            l2_loss += tf.nn.l2_loss(b1) 
            #output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)
            output1 = Dense(256, activation='relu', name="output1")(output0)  # w1과 b1을 합친 Dense 레이어
            
            #w2 = tf.get_variable("w2", shape=[256,128], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.Variable(GlorotUniform()(shape=[256, 128]), name="w2")
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2") 
            l2_loss += tf.nn.l2_loss(w2) 
            l2_loss += tf.nn.l2_loss(b2) 
            #output2 = tf.nn.relu(tf.matmul(output1, w2) + b2) 
            output2 = Dense(128, activation='relu', name="output2")(output1)  # w2와 b2를 합친 Dense 레이어

            
            #w = tf.get_variable("w", shape=(128, 2), initializer=tf.contrib.layers.xavier_initializer()) 
            w = tf.Variable(GlorotUniform()(shape=(128, 2)), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b") 
            l2_loss += tf.nn.l2_loss(w) 
            l2_loss += tf.nn.l2_loss(b) 
            
            #self.scores = tf.nn.xw_plus_b(output2, w, b, name="scores") 
            #self.scores = tf.linalg.matmul(output2, w) + b
            # 최종 점수 계산을 위한 Dense 레이어
            self.scores = Dense(2, activation=None, name="scores")(output2)  # w와 b를 합친 Dense 레이어
            #self.predictions = tf.argmax(self.scores, 1, name="predictions") 
            self.predictions = Lambda(lambda x: tf.argmax(x, axis=1), name="predictions")(self.scores)

        with tf.name_scope("loss"): 
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y) 
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"): 
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1)) 
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy") 
