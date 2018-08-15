---
layout: post
title: Optimization Modules - A Brief Overview
---

In this blog post, I would like to give a brief overview of the existing gradient descent optimization algorithms that are available. There are lots of good resources available online. You can check them at the References section at the end of this post.

The existing TMVA submodule has always used gradient descent to update the parameters and minimize the cost of the neural networks. More advanced optimization methods can speed up learning and perhaps even get you to a better final value for the cost function. Having a good optimization algorithm can be the difference between waiting days vs. just a few hours to get a good result.

**Gradient descent** is a **first-order iterative optimization algorithm** for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.
Gradient descent goes "downhill" on a cost function `J`. Think of it as trying to do this:

<br>
![Gradient Descent]({{ site.baseurl }}/images/gradientdescent.jpg)
<br>


## Gradient Descent Variants:

There are three variants of gradient descent, which depends on how much data you use to cacluate the gradients and perform an update. They are as follows,

### 1) Batch Gradient Descent: 
Vanilla gradient descent, also known as batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters $$\theta$$ for the entire training dataset. It supports maximum vectorization, but if the data is large, it cannot fit into the memory.

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta)$$

### 2) Stochastic Gradient Descent:
Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example of the training dataset. It cannot exploit vectorization, since it has to iterate through all the training examples and make an update for each training example. It also shows a lot of fluctuations before converging to the solution.

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)})$$

### 3) Mini-Batch Gradient Descent:

Mini-batch gradient descent finally takes the best of both approaches and performs an update for every mini-batch of n training examples. The size of the mini-batch is usually in the power of 2 like 64 or 256, but can vary depending on the applications. It exploits vectorization to some extent and its update is also fast. It is the most preferred way of update among these variants.

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})$$


## Gradient descent optimization algorithms:

Here, I'll discuss about the various gradient descent optimization algorithms that are proven to work best in most of the applications.

### 1) Momentum based update:

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in the below image. It does this by adding a fraction $$\gamma$$ of the update vector of the past time step to the current update vector.

**SGD without Momentum:** ![SGD without momentum]({{ site.baseurl }}/images/without_momentum.gif)
**SGD with Momentum:** ![SGD with momentum]({{ site.baseurl }}/images/with_momentum.gif) 

The momentum update is done as follows,

$$
\begin{align}
\begin{split}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\  
\theta &= \theta - v_t
\end{split}
\end{align}
$$

The usual value of $$\gamma$$ is 0.9. ie ( $$\gamma$$ < 1 )

### 2) Nesterov accelerated Momentum:

Ilya Sutskever suggested a new form of momentum that often works better. It is inspired by the nesterov method for optimizing convex functions. First, make a big jump in the direction of the previous accumulated gradient. Then, measure the gradient where you end up and make correction. Its better to correct a mistake after you have made it. :P

**Nesterov Update:**
![Nesterov Update]({{ site.baseurl }}/images/nesterov_update.png)

Here, <span style="color:brown">brown vector = jump</span>, <span style="color:red">red vector = correction</span>, <span style="color:green">green vector =  accumulated gradient</span>, <span style="color:blue">blue vector = standard momentum</span>.

The Nesterov update is done as follows,

$$
\begin{align}
\begin{split}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\  
\theta &= \theta - v_t
\end{split}
\end{align}
$$

The usual value of $$\gamma$$ is 0.9. ie ( $$\gamma$$ < 1 ) and it depends on the application.

### 3) Adagrad:

AdaGrad is an optimization method that allows different step sizes for different features. It increases the influence of rare but informative features i.e. It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing with sparse data.

The Adagrad update is done as follows,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}
$$

where,<br>
    $$G_{t}$$ - Sum of the squares of the past gradients w.r.t all parameters $$\theta$$ along its diagonal.<br>
    $$\odot$$ - Matrix-vector dot product.<br>
    $$g_{t}$$ - Gradient at time step $$t$$.<br>
    $$\eta$$ - Learning rate.<br>
    $$\epsilon$$ - Smoothing term that avoids division by zero and is usually of the order of $$1e-8$$.<br>

### 4) Adadelta:

Adadelta is an extension of Adagrad that tries to reduce the monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta stores only the window of accumulated past gradients to some fixed window size $$w$$.

And Instead of storing all the past gradients of window size w, it stores the decaying average of the past squared gradients. The running average $$E[g^2]_t$$ at time step $$t$$ is calculated as,

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g^2_t$$

The root mean squared (RMS) error of the gradient is therefore,

$$RMS[g]_{t} = \sqrt{E[g^2]_t + \epsilon}$$

And also the decaying average of the past squared updates is computed as,

$$E[\Delta \theta^2]_t = \gamma E[\Delta \theta^2]_{t-1} + (1 - \gamma) \Delta \theta^2_t$$

The root mean squared (RMS) error of the updates is therefore,

$$RMS[\Delta \theta]_{t} = \sqrt{E[\Delta \theta^2]_t + \epsilon}$$

Since $$RMS[\Delta \theta]_{t}$$ is unknown, we approximate it with the RMS of parameter updates until the previous time step i.e. $$RMS[\Delta \theta]_{t-1}$$

Thus, the Adadelta update is done as follows,

$$
\begin{align}
\begin{split}
\Delta \theta_t &= - \dfrac{RMS[\Delta \theta]_{t-1}}{RMS[g]_{t}} g_{t} \\
\theta_{t+1} &= \theta_t + \Delta \theta_t 
\end{split}
\end{align}
$$

Here, the parameters usually take the default value,<br>
$$\gamma$$ - Usually around 0.9.<br>
$$\epsilon$$ - Smoothing term that avoids division by zero and is usually of the order of $$1e-8$$.<br>

### 5) RMSprop:

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton. The main idea is **"Divide the gradient by a running average of its recent magnitude"**. It is similar to Adadelta but it is developed independently to overcome the disadvantages of the Adagrad algorithm.

The RMSprop update is done as follows,

$$
\begin{align}
\begin{split}
E[g^2]_t &= \gamma E[g^2]_{t-1} + (1-\gamma) g^2_t \\  
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}
\end{split}
\end{align}
$$

where,<br>
$$\gamma$$ - Usually around 0.9.<br>
$$\eta$$ - Learning rate, usually around 0.001.<br>
$$E[g^2]_t$$ - Decaying average of the past squared gradients at time step $$t$$.<br>

### 6) Adam:

Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. It stores both the decaying average of the past gradients $$m_t$$, similar to momentum and also the decaying average of the past squared gradients $$v_t$$, similar to RMSprop and Adadelta. Thus, it combines the advantages of both the methods. Adam is the default choice of the optimizer for any application in general.

The decaying average of the past gradients $$m_t$$ and the past squared gradients $$v_t$$ is computed as follows,

$$
\begin{align}
\begin{split}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\  
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2  
\end{split}
\end{align}
$$

And since these $$m_t$$ and $$v_t$$ are initialized with zeros, they are biased towards zero, especially during the initial time steps. Thus, to avoid these biases, the bias corrected versions of them are computed as follows,

$$
\begin{align}
\begin{split}
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2} \end{split}
\end{align}
$$

Thus, the Adam update is as follows,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

where,<br>
$$\beta_1$$ - usually 0.9.<br>
$$\beta_2$$ - usually 0.999.<br>
$$\eta$$ - Learning rate.<br>
$$\epsilon$$ - usually of the order of $$1e-8$$.<br>

### 7) Adamax:

Adamax is the generalization of the Adam algorithm to the $$\ell_{\infty}$$ norm. Kingma and Ba show that $$v_t$$ with $$\ell_{\infty}$$ converges to the more stable value.

The infinity norm-constrained $$v_t$$ is denoted as $$u_t$$ and is computed as follows,

$$
\begin{align}
\begin{split}
u_t &= \beta_2^\infty v_{t-1} + (1 - \beta_2^\infty) |g_t|^\infty\\  
              & = \max(\beta_2 \cdot v_{t-1}, |g_t|)
\end{split}
\end{align}
$$

Here, since $$u_t$$ relies on the max operation, it is not biased towards zero unlike the ones in the Adam algorithm.

Thus, the Adamax update is as follows,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{u_t} \hat{m}_t
$$

where,<br>
$$\beta_1$$ - usually 0.9.<br>
$$\beta_2$$ - usually 0.999.<br>
$$\eta$$ - Learning rate, usually 0.002.<br>

### 8) Nadam:

Nadam is similar to Adam which is a combination of the momentum and the RMSprop. Nadam can be viewed as a combination of the nesterov accelerated momentum and the RMSprop. Here, we do not need to modify the $$\hat{v}_t$$. The momentum vector equations are as follows,

$$
\begin{align} 
\begin{split}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t\\  
\hat{m}_t & = \frac{m_t}{1 - \beta^t_1}\\
\theta_{t+1} &= \theta_{t} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{split}
\end{align}
$$

Expanding the last equation gives,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\dfrac{\beta_1 m_{t-1}}{1 - \beta^t_1} + \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1})
$$

But since, 

$$ \hat{m}_{t-1} =  \dfrac{m_{t-1}}{1 - \beta^t_1}$$

we get,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\beta_1 \hat{m}_{t-1} + \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1})
$$

We can now add Nesterov momentum just as we did previously by simply replacing this bias-corrected estimate of the momentum vector of the previous time step $$\hat{m}_{t−1}$$ with the bias-corrected estimate of the current momentum vector $$\hat{m}_t$$.

Thus, the Nadam update is as follows,

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\beta_1 \hat{m}_t + \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1})
$$

### Summary of the various update equations :
<br>

![Update Equations]({{ site.baseurl }}/images/update_eqn.jpg)

### References:

1) [An overview of gradient descent optimization algorithms - Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html)

2) [Difference between Batch Gradient Descent and Stochastic Gradient Descent](https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1)

3) [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

4) [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

5) [AdaDelta: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

6) [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)

7) [Nadam: Nesterov Adam optimizer](http://cs229.stanford.edu/proj2015/054_report.pdf)

8) [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)

9) [Keras Optimizers](https://keras.io/optimizers/)
