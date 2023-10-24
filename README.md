# VON
Versatile Ordering Network (VON) can be used across tasks, objectives, and data distributions. By leveraging the power of reinforcement learning and a greedy rollout strategy, the network can automatically learn ordering strategies to improve itself.

# Paper
Reviewing

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

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
