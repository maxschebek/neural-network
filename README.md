# Neural network with backpropagation

This repo contains an implemenation of a neural network utilizing numpy. The implementation follows Andrew Ng's [machine learning course](https://www.coursera.org/learn/machine-learning) .


## Forward propagation
We consider a network consisting of $L$ layers: one input layer, one output layer, and $L$-$2$ hidden layers.  A given layer $j$ of the network with $s_j$ units can be represented in terms of two vectors, $\mathbf{z}^{(j)}$ and $\mathbf{a}^{(j)}$, whereas the propagation of data from the input to the output layers is given in terms of $L$-$1$ weight matrices $\boldsymbol{\Theta}$.

For a given input instance with features $\mathbf{x}$, for the first layer ($j=1$) the vector $\mathbf{a}$ is set as

$$
\mathbf{a}^{(1)} = (1,\mathbf{x})^{\rm T}.
$$

For $L > j > 1$, $\mathbf{z}^{(j)}$ and $\mathbf{a}^{(j)}$ can be obtained following

$$
\mathbf{z}^{(j)} = \boldsymbol{\Theta}^{(j-1)}a^{(j-1)}
$$
with

$$
\mathbf{a}^{(j)} = \left(1,g(\mathbf{z}^{(j)})\right)^{\rm T}=\left(1, g(z^{(j)}_1),g(z^{(j)}_2),...,g(z^{(j)}_{s_j})\right)^{\rm T}.
$$
The activation function $g$ is given by the sigmoid function

$$
g(z) = \frac{e^z}{1+ e^z}.
$$

The last layer does not have a bias unit, such that the output layer is given by

$$
\mathbf{a}^{(L)} = g(\mathbf{z}^{(L)}) \equiv \hat{\mathbf{y}}.
$$
## Cost function
For $m$ instances in the training set, the regularized cost function of the network is defined as 

$$
J(\{\boldsymbol{\Theta_i}\}) = -\frac{1}{m}\sum_m\mathbf{y}_m\cdot\log(\hat{\mathbf{y}_m}) - (1 -\mathbf{y}_m)\cdot\log(1 -\mathbf{\hat{y}}_m) + \frac{\lambda}{2m}\sum_j\sum_{i=2}\sum_{l=1}\left(\Theta^{(j)}_{il}\right)^2
$$

## Backpropagation

Given a training instance $\mathbf{y}$, the error of the last layer ($j=L$) is set to

$$
\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}.
$$

For $L > j >1$, the error associated to each layer is given by

$$
\boldsymbol{\delta}^{(j)} = \left(\boldsymbol{\tilde{\Theta}}^{(j)}\right)^{\rm T} \delta^{(j+1)} \cdot g'(z^{(j)}),
$$

where $\cdot$ denotes the scalar product and $\boldsymbol{\tilde{\Theta}}^{(j)}$ corresponds to the weight matrix $j$ without the first column. This is needed to exclude the bias unit from the backpropagation, which is not connected to the input unit.

The error associated with the weight matrix $j$ is given by a matrix $\mathbf{D}^{(j)}$, which is defined as

$$
D^{(j)}_{il} = \frac{\partial J}{\partial \Theta^{(j)}_{il}}.
$$

Using the error of each individual layer, the error associated with the weight matrix $j$ can be defined as

$$
D^{(j)}_{il} = \begin{cases} \frac{1}{m} \Delta^{(j)}_{il} \qquad \text{  if } i=0 \\ \frac{1}{m} \Delta^{(j)}_{il} + \frac{\lambda}{m} \Theta^{(j)}_{il} \qquad \text{else}.
\end{cases}
$$
where the matrix $\boldsymbol{\Delta}^{(j)}$ accumulates the errors when going through the training set with $m$ instances

$$
\boldsymbol{\Delta}^{(j)} = \boldsymbol{\Delta}^{(j)} + \boldsymbol{\delta}^{(j+1)}\left(\mathbf{a}^{(j)}\right)^{\rm T}.
$$