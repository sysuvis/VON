# VON
Versatile Ordering Network (VON) can be used across tasks, objectives, and data distributions. By leveraging the power of reinforcement learning and a greedy rollout strategy, the network can automatically learn ordering strategies to improve itself.

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Files

* data: Including the CIFAR-10 dataset and the sch dataset that we sampled and cleaned.
* demo: All the code of demo to test the ordering across scale and metric in CIFAR-10 data.
* nets: The model implementation of VON.
* pretrained: Including the trained model parameters, that you can use in testing.
* problems: The task definition of ordering. Including the dataset inputting, metric definition and related tools.
* utils: The tools for training and testing.
* eval.py: The testing code.
* flaskfordemo.py: The server for demo.
* options.py: The settings for training.
* reinforce_baselines.py: The baselines of reinforcement.
* run.py & train.py: The implementations of training.

## Usage

### How to input your data

1. Prepare your data to the size: [num_sample, num_points, dim]. The 'num_sample' is the sampling number. The 'num_points' is the points amount. The 'dim' is the dimension of your embedding.
2. Change the 'node_dim' to your need in nets/attention_model.py.
4. Implement your code at '__init__' of problems/order/problem_order.py. Here is an example for data adding:
```commandline
...
elif mission == 'dblp':
       if mode == 'train':
              with open('data/DBLP/dblp_50_dis_mix_train_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
              self.data = data_tsne
       elif mode == 'test':
              with open('data/DBLP/dblp_100_dis_g_o_tsne_test.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
              self.data = [data_tsne, data_tsne]
       else:
              print('Please input right run mode!')
...
```
Note: The nan and inf are not allowed in data.

### How to use your metric
1. Implement your metric by any way in python.
2. Add them into 'get_costs' of problems/order/problem_order.py. Here is an example:
```commandline
def moransi(d):
       ...
       return ...
...
elif cost_choose == 'moransI':
       ret_res = torch.zeros(d.size(0))
       for a in range(d.size(0)):
              ret_res[a] = 1- moransi(d[a, :, :])
       return ret_res, None
```

### Training
Options:
--graph_size: >=2, eg. 20, 50, 100...;
--run_name: any words are permitted;
--mission: 'CIFAR10', 'fashionmnist';
--cost_choose: TSP, stress, moransI.

For model, you can change the options 'model' in the 'second_layer_decoder' of nets/attention_model.py. You can choose 'm', 'a', 'c' and 'AM'.

eg.
```commandline
python run.py --graph_size 50 --baseline rollout --run_name 'test' --mission 'CIFAR10' --cost_choose stress
```
### Testing
Options:
--mission: 'CIFAR10', 'fashionmnist', 'demo';
--size: >=2, eg. 20, 50, 100...;
--model: The model lib;
--dataset_number and --decode_strategy are no need to change.

1. Move the trained model to target file(eg. pretrained/)
2. Run: eg.``` python eval.py --model 'pretrained/TSP' --decode_strategy 'greedy' -f --run_mode 'test' --mission 'demo' --dataset_number 1 --size 50 ```
All the options of command can be replaced following your needs.

### With demo
1.```python .\flaskfordemo.py```
2. Extract the zip package in the demo/cifar10 folder. Then, double click 'panel.html' in the /demo to start.
3. Choose 'TSP' as the loss.
4. Brush points in the scatter. And wait seconds.
5. Then you can freely explore. 
