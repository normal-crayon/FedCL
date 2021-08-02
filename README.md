# FedCL
## phase 1:
A simple illustrative project on combination of federated and continual learning using avalanche and flower frameworks
FedCL - contains two clients with MNIST dataset split in half with first half of classes (0, 1, 2, 3, 4) in one client and the other half in another client. This demonstartes class and task
increamental CL with the classes split into different clients. 

## phase 2:
Added CIFAR10 clients (client3.py and client4.py) with classes split in half with clients 3 and 4. 

## Usage:
1. navigate to FedCL directory, start server.py first

<pre><code>python server.py</code></pre>

2. start any two clients (minimum two clients for federated learning to start)

<pre><code>python client1.py</code></pre>

3. for visualization use tensorboard 

<pre><code>tensorboard --logdir tb_data</code></pre>

