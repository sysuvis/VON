# VON
Versatile Ordering Network (VON) can be used across tasks, objectives, and data distributions. By leveraging the power of reinforcement learning and a greedy rollout strategy, the network can automatically learn ordering strategies to improve itself.

# Paper
Revising

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

### With demo
```commandline
1. python .\flaskfordemo.py
```
```commandline
2. Extract the zip package in the demo/cifar10 folder. Then, double click 'panel.html' in the /demo to start.
```
```commandline
3. Choose 'TSP' as the loss.
```
```commandline
4. Brush points in the scatter. And wait seconds.
```
```commandline
5. Then you can freely explore.
```

### Training

```commandline
python run.py --graph_size $size$ --baseline rollout --run_name $name$ --mission $data$ --cost_choose $loss$
eg. python run.py --graph_size 50 --baseline rollout --run_name 'test' --mission 'CIFAR10' --cost_choose stress
```
### Testing

```commandline
1. Move the trained model to target file(eg. pretrained/)
```
```commandline
2. Run: eg. "python eval.py --model 'pretrained/TSP' --decode_strategy 'greedy' -f --run_mode 'test' --mission 'demo' --dataset_number 1 --size 50"
```
All the options of command can be replaced following your needs.

* For loss, we implement the TSP, Stress, Moran's I, LA, BW, PR, Symmetry and Correlation.
* For model, you can change the options 'model' in the 'second_layer_decoder' of nets/attention_model.py.
* You can replace the default data to yours. First, prepare your data to the size: [num_sample, num_points, dim]. Second, change the dim to your need in nets/attention_model.py. Third, add your data at '__init__' of problems/order/problem_order.py.
* You can add the loss at 'get_costs' of problems/order/problem_order.py.

       
