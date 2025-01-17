import numpy as np
from multiprocessing import Process, Queue
import random

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

#从user_train中随机选择一个用户，生成一个长度为maxlen的序列，其中最后一个物品为正样本，其余物品为负样本
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, threshold_user, threshold_item,result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            if random.random() > threshold_item:#zengjiade
                i = np.random.randint(1, itemnum + 1)
                nxt = np.random.randint(1, itemnum + 1)
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        if random.random() > threshold_user:#random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0zengjiade
            user = np.random.randint(1, usernum + 1)

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))#返回值为(user, seq, pos, neg)：一个四元组，分别为用户id、物品序列、正样本id、负样本id。

#初始化WarpSampler类，创建多个进程，并启动采样函数
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, threshold_user=0.08, threshold_item=0.99, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      threshold_user,  # zengjiade
                                                      threshold_item,  # zengjiade
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):#从队列中获取一个batch的数据
        return self.result_queue.get()

    def close(self):#关闭所有进程，释放资源
        for p in self.processors:
            p.terminate()
            p.join()
