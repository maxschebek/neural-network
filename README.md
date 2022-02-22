# Neural network with backpropagation

<div align="center">
  
![download](https://user-images.githubusercontent.com/33028448/155228717-3d6a1b50-d4ec-4c5a-9540-c76c826cdf7b.png)
![download (1)](https://user-images.githubusercontent.com/33028448/155228713-ebc89c14-f2f5-4321-bcf4-152f04a0736c.png)

</div>

This repo contains an implemenation of a neural network utilizing linear algebra routines provided by [numpy](https://numpy.org/doc/stable/index.html). The implementation follows Andrew Ng's [machine learning course](https://www.coursera.org/learn/machine-learning). Details  can be found in the [documentation](documentation/neural_network.pdf).


## Setup

Install poetry

```console
pip install poetry
```

Change into the directory of the repository and run 

```console
poetry install
```

to create the virtual environment and install all dependencies. Run

```console
poetry shell
```

to activate the virtual environment.

Tests can be run using 


```console
poetry run pytest
```
