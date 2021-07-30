#!/usr/bin/env python
 
import sys
import torch.nn as nn
import torch
import torchvision
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim

from avalanche.benchmarks.utils import avalanche_dataset
from avalanche.benchmarks.classic.ccifar10 import  SplitCIFAR10
from avalanche.training.strategies import EWC, Naive, Replay
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import TensorboardLogger, InteractiveLogger

import flwr as fl

from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cifar_transform = svhn_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Grayscale(num_output_channels=1),
          torchvision.transforms.Resize((28, 28))
          
])

# ids = ["airplane", "automobile", "bird", "cat", "deer"]
ids = [5, 6, 7, 8, 9]
split_cifar = SplitCIFAR10(n_experiences=5, fixed_class_order=ids, return_task_id=True, shuffle=True, train_transform=cifar_transform, eval_transform=cifar_transform)

train_stream = split_cifar.train_stream
test_stream = split_cifar.test_stream

full_cifar = SplitCIFAR10(n_experiences=5, return_task_id=True, shuffle=True, eval_transform=svhn_transform)
full_test_stream = full_cifar.test_stream

for exp in train_stream:
    t = exp.task_label
    exp_id = exp.current_experience
    task_train_ds = exp.dataset
    print('Task {} batch {} -> train'.format(t, exp_id))
    print('This batch contains', len(task_train_ds), 'patterns')


for exp in full_test_stream:
    t = exp.task_label
    exp_id = exp.current_experience
    task_train_ds = exp.dataset
    print('Task {} batch {} -> test'.format(t, exp_id))
    print('This batch contains', len(task_train_ds), 'patterns')

tb2_logger = TensorboardLogger()
int_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, stream=True),
    loss_metrics(epoch=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[tb2_logger, int_logger],
    benchmark=split_cifar,

)

class MnistCNN(nn.Module):

    def __init__(self):
        super(MnistCNN, self).__init__() 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model = MnistCNN()

opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

naive_strategy = Naive(
    model, opt, criterion, train_mb_size=100, train_epochs=5,
    eval_mb_size=100, device=device, evaluator=eval_plugin
)

replay_strategy =  Replay(
    model, opt, criterion, train_mb_size=100, train_epochs=5,
    eval_mb_size=100, device=device, evaluator=eval_plugin
)

ewc_strategy = EWC(
    model, opt, criterion, ewc_lambda=0.4, train_mb_size=100, train_epochs=5, keep_importance_data=True,
    eval_mb_size=100, device=device, evaluator=eval_plugin
)

def train(model, train_stream, strategy):
    for train_task in train_stream:
        data = train_task.dataset
        strategy.train(train_task)
        print(strategy.eval(test_stream))
    return len(data)

def test(model, test_stream, strategy):
    results = []
    for test_task in test_stream:
        data = test_task.dataset
        results.append(strategy.eval(test_stream))
        print(results)
        return len(data), results


class MnistClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _,val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        data = train(model=model, train_stream=train_stream, strategy=ewc_strategy)
        return self.get_parameters(), data, {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        data, results = test(model=model, test_stream=full_test_stream, strategy=ewc_strategy)
        return float(results[0]['Loss_Stream/eval_phase/test_stream/Task004']), \
             data, {'accuracy':results[0]['Top1_Acc_Stream/eval_phase/test_stream/Task004']}


if __name__ == "__main__":    
    fl.client.start_numpy_client("localhost:5000", client=MnistClient())
  