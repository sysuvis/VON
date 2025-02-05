
# VON
Versatile Ordering Network (VON) can be used across tasks, objectives, and data distributions. By leveraging the power of reinforcement learning and a greedy rollout strategy, the network can automatically learn ordering strategies to improve itself.

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* torchvision
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)
* gevent
* flask
* flask_cors
(In general, the following dependencies are built-in to Python>=3.8)
* os
* pickle
* json
* math
* argparse

When pytorch=2.6, `torch.load` may encounter an error when called:
_pickle.UnpicklingError: Weights only load failed. WeightsUnpickler error: Unsupported global: GLOBAL nets.attention_model.AttentionModel was not an allowed global by default.
The reason is pytorch2.6 changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. The latest version fixed this issue by adding the set of `weights_only` in the use of `torch.load` in utils/function.py.

## Files
### folders
* **data**: The CIFAR-10 dataset and the SCH dataset that we sampled and cleaned.
* **demo**: The demo for interactive ordering across scale and metric using the CIFAR-10 dataset.
* **nets**: The model implementation of VON.
* **pretrained**: The models with trained parameters, that you can use in testing.
* **problems**: The definition of ordering task, including the dataset (input), metric definition, and other utility functions.
* **utils**: The utility functions for training, testing, logging, saving files, and loading parameters.

### python sources
* **eval.py**: The code for inferencing.
* **run.py & train.py**: The code for training. You should run ```python run.py``` during training.
* **flaskfordemo.py**: The server for the interactive demo.
* **options.py**: The settings for training.
* **reinforce_baselines.py**: The baselines of reinforcement.
* **plot_box.py**: The script to plot the Fig. 10 in paper.
* The models trained by the metrics from specialized methods of VON-m for Fashion-MNIST, MNIST, ImageNet, and CIFAR-10 are in https://drive.google.com/file/d/14aovO_FJXC9gyuINV4tLiEC5ZgYpI4M-/view?usp=drive_link. After extracting, you can find the corresponding model based on the folder name. For example, "CF_dpq" represents the model trained on the CIFAR-10 dataset using DPQ as the metric. The test datasets are in: https://drive.google.com/file/d/1h93Jr8P2caGq3cpMMp5WTNn1kWRwbucv/view?usp=drive_link.


## Usage

### How to input your data

1. Prepare your data as a three-dimensional tensor in the shape of: ```[num_sample, num_points, dim]```. The ```num_sample``` is the sampling number, the ```num_points``` is the points amount, and the ```dim``` is the dimension of your embedding.
2. Please note that ```dim``` is the dimension of the input data, to modify the encoding dimension of the network you may use the ```node_dim``` in **nets/attention_model.py**.
3. Implement your code in the ```__init__``` function in **problems/order/problem_order.py**. Here is an example for data adding:
```commandline
...
elif dataset == 'dblp':
	if mode == 'train':
	    with open('data/DBLP/dblp_50_dis_mix_train_tsne.pkl', 'rb') as f1:
			coordinates = pickle.load(f1)
			self.data = coordinates 
	elif mode == 'test':
		with open('data/DBLP/dblp_100_dis_g_o_tsne_test.pkl', 'rb') as f1:
			data_tsne = pickle.load(f1)
			self.data = [coordinates, coordinates]
	else:
		print('Please input right run mode!')
...
```
**Note**: Please make sure your data is clean, e.g., without nan and inf.

### How to use customized metric
1. Implement your metric as a function in python, for example:
```commandline
def moransi(d):
       ...
       return ...
```

2. To make the metric available in the option list, you may add the metric into the ```get_costs``` function in **problems/order/problem_order.py**. Here is an example:
```commandline
...
elif metric == 'moransI':
       ret_res = torch.zeros(d.size(0))
       for a in range(d.size(0)):
              ret_res[a] = 1 - moransi(d[a, :, :])
       return ret_res, None
```
**Note**: You may need to convert the metric (larges is better) to a loss (smaller is better). Here we use ```1 - moransi(d[a, :, :])```.

### Training
Options:
--```mission```: the dataset, e.g.,  'CIFAR10', 'fashionmnist';
--```metric```: the metric, e.g., 'TSP', 'stress', 'moransI';
--```graph_size```: the number of points to be ordered (>=2), e.g. 20, 50, 100...;
--```run_name```: a folder name for saving the models during training (any words are permitted).

```commandline
python run.py --graph_size 50 --run_name test --mission FM --metric tsp
```
You can try your settings to train your model by this command.

### Testing
Options:
--```model```: the path of model, e.g., 'pretrained/test';
--```run_mode```: the mode set, a.g., 'test';
```commandline
python eval.py --model pretrained/test -f --run_mode test --mission FM
```
After training, you can test your model by this command.

### Inferencing
Options:
--```dataset```: the dataset, e.g.,  'CIFAR10', 'fashionmnist';
--```sample_size```: the number of points to be ordered (>=2), e.g. 20, 50, 100...;
--```model```: the trained model used for ordering.

1. Move the trained model to target file(eg. pretrained/)
2. Run: eg.``` python eval.py --model 'pretrained/CIFAR-TSP' --run_mode 'test' --dataset 'CIFAR10' --sample_size 50 ```
All the options of command can be replaced following your needs.

### Running the interactive demo
** Before starting, please confirm that your environment contains all of the dependencies mentioned above.

** The default version is for torch on GPU. If your environment is CPU, please replace the problems/order/problem_order.py with problems/order/problem_order_cpu.py.

1. Start the server for the demo using ```python .\flaskfordemo.py```.
2. Extract the zip package in the **demo/cifar10** folder, then make sure all the extracted pictures' path like: **demo/cifar10/images256/cf10_image_xxx.jpg**. (Some decompression methods add an extra subfolder, e.g. **demo/cifar10/images256/images256/cf10_image_xxx.jpg**, which may cause file reading errors.)
3. Use a browser to open **panel.html** in the **/demo** folder to start the front end.
4. Choose a metric in the dropdown list in the top left corner, e.g., 'Moran's I', 'TSP'. The default is 'TSP'.
5. Brush points in the scatter plot on the left, and view the ordered images on the right.
   
This demo is a quick way to test the performance of VON. You can reproduce the figure 2 in the appendix of our paper by brushing the scatter plot in the same area.
