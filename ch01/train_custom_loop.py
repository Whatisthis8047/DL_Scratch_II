import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

# 1. Set hyperparameter
max_epoch = 100
hidden_size = 10
batch_size = 30
learning_rate = 0.1

# 2. Load data, initialize model(Two Layer Net), Optimizer
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(learning_rate)

# 3. set variables for learning
data_size = len(x)
max_iter = data_size//batch_size
total_loss, loss_count = 0, 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iter):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 4. calculate gradient
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 5. print learning status
        if (iters+1) % 10 == 0:
            avg_loss = round(total_loss/loss_count,3)
            print(f'| epoch: {epoch+1} | iters: {iters+1} | loss: {avg_loss}')
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0