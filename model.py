from modules import *
import tensorflow.compat.v1 as tf


class Model():
    def __init__(self, usernum, itemnum, args, reuse=tf.AUTO_REUSE):
        tf.compat.v1.disable_eager_execution()
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.u = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.input_seq, 0)), -1)

        src_masks = tf.math.equal(self.input_seq, 0)

        with tf.compat.v1.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.item_hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.item_hidden_units + args.user_hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            u0_latent, user_emb_table = embedding(self.u[0],
                                                  vocab_size=usernum + 1,
                                                  num_units=args.user_hidden_units,
                                                  zero_pad=False,
                                                  scale=True,
                                                  l2_reg=args.l2_emb,
                                                  scope="user_embeddings",
                                                  with_t=True,
                                                  reuse=reuse
                                                  )
            u_latent = embedding(self.u,
                                 vocab_size=usernum + 1,
                                 num_units=args.user_hidden_units,
                                 zero_pad=False,
                                 scale=True,
                                 l2_reg=args.l2_emb,
                                 scope="user_embeddings",
                                 with_t=False,
                                 reuse=reuse
                                 )
            # Dropout使用tf.layers.dropout函数对self.seq进行dropout操作，并设置dropout率为args.dropout_rate。同时，使用tf.convert_to_tensor将self.is_training转换为Tensor类型，并作为参数传入，以便在训练和测试时使用不同的dropout策略
            self.u_latent = tf.tile(tf.expand_dims(u_latent, 1), [1, tf.shape(self.input_seq)[1], 1])
            self.hidden_units = args.item_hidden_units + args.user_hidden_units
            self.seq = tf.reshape(tf.concat([self.seq, self.u_latent], 2),
                                  [tf.shape(self.input_seq)[0], -1, self.hidden_units])
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   values=self.seq,
                                                   key_masks=src_masks,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units])
                    self.seq *= mask
            self.seq = normalize(self.seq)
        #print(np.shape(pos))
        user_emb = tf.reshape(self.u_latent, [tf.shape(self.input_seq)[0] * args.maxlen,
                                              args.user_hidden_units])
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        #print(np.shape(pos_emb))
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units])
        print(np.shape(seq_emb))#（？，128）
        test_item_emb = item_emb_table
        test_user_emb = tf.tile(tf.expand_dims(u0_latent, 0), [itemnum + 1, 1])
        test_item_emb = tf.reshape(tf.concat([test_item_emb, test_user_emb], 1), [-1, self.hidden_units])
        
        print(np.shape(test_item_emb))#（12102，128）
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        print(np.shape(self.test_logits))#（？，12102）
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum+1])
        #print(np.shape(self.test_logits))#(?,100,12102)
        self.test_logits = self.test_logits[:, -1, :]
        print(np.shape(self.test_logits))#（？，12102）
        # prediction layer
        #print(np.shape(pos_emb),np.shape(seq_emb))
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        #print(np.shape(self.pos_logits))
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is tf.AUTO_REUSE:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.is_training: False})
