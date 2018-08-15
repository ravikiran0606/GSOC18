---
layout: post
title: Adam and RMSProp Optimizer - Implementation and Testing:-
---

In this blog post, I'll be explaining the implementation of the Adam Optimizer, RMSProp optimizer with and without momentum approach. 

## RMSProp Optimizer:

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton. The main idea is **"Divide the gradient by a running average of its recent magnitude"**. It is similar to Adadelta but it is developed independently to overcome the disadvantages of the Adagrad algorithm.

Thus, the update is implemented as follows, ( similar to the tensorflow implementation )

```
Vt = rho * Vt-1 + (1-rho) * currentSquaredGradients
Wt = momentum * Wt-1 + (learningRate * currentGradients) / (sqrt(Vt + epsilon))
theta = theta - Wt
```

So, one step of update is performed as,

$$
\begin{align}
\begin{split}
v_t &= \rho v_{t-1} + (1-\rho) \nabla_\theta^2 J( \theta) \\
w_t &= \gamma w_{t-1} + \dfrac{\eta}{\sqrt{v_t + \epsilon}} \nabla_\theta J( \theta)  \\  
\theta &= \theta - w_t
\end{split}
\end{align}
$$

## Testing RMSProp:

I used the same unit tests approach as for SGD optimizer. Have a look at **Testing the SGD optimizer post**.

<div>
    <a href="https://plot.ly/~ravikiran0606/36/?share_key=dQ2PnWziGGbXA8mVVRDxK0" target="_blank" title="RMSPROPUTP" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/36.png?share_key=dQ2PnWziGGbXA8mVVRDxK0" alt="RMSPROPUTP" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:36" sharekey-plotly="dQ2PnWziGGbXA8mVVRDxK0" src="https://plot.ly/embed.js" async></script>
</div>

<div>
    <a href="https://plot.ly/~ravikiran0606/37/?share_key=sNfftgUs2x2JITp0XKOhDO" target="_blank" title="RMSPROPMUTP" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/37.png?share_key=sNfftgUs2x2JITp0XKOhDO" alt="RMSPROPMUTP" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:37" sharekey-plotly="sNfftgUs2x2JITp0XKOhDO" src="https://plot.ly/embed.js" async></script>
</div>

The above figures shows the convergence of the training and testing errors for the RMSProp optimizer without and with momentum during the unit tests.

## Adam Optimizer:

Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter. It stores both the decaying average of the past gradients $$m_t$$, similar to momentum and also the decaying average of the past squared gradients $$v_t$$, similar to RMSprop and Adadelta. Thus, it combines the advantages of both the methods. Adam is the default choice of the optimizer for any application in general.

Thus, the update is implemented as follows, ( similar to the tensorflow implementation )

```
Mt = beta1 * Mt-1 + (1-beta1) * currentGradients
Vt = beta2 * Vt-1 + (1-beta2) * currentSquaredGradients
alpha = learningRate * sqrt(1 - beta2^t) / (1-beta1^t)
theta = theta - alpha * Mt / (sqrt(Vt) + epsilon)
```

So, one step of update is performed as,

$$
\begin{align}
\begin{split}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J( \theta) \\ 
v_t &= \beta_2 v_{t-1} + (1-\beta_2) \nabla_\theta^2 J( \theta) \\ 
\alpha &= \eta \dfrac{\sqrt{(1-\beta_2^t)}}{(1-\beta_1^t)} \\
\theta &= \theta - \alpha \dfrac{m_t}{\sqrt{v_t}+ \epsilon}
\end{split}
\end{align}
$$

## Testing Adam:

I used the same unit tests approach as for SGD optimizer. Have a look at **Testing the SGD optimizer post**.

<div>
    <a href="https://plot.ly/~ravikiran0606/35/?share_key=adux2BQhLVIq0OPUVU3pO0" target="_blank" title="ADAMUTP" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/35.png?share_key=adux2BQhLVIq0OPUVU3pO0" alt="ADAMUTP" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:35" sharekey-plotly="adux2BQhLVIq0OPUVU3pO0" src="https://plot.ly/embed.js" async></script>
</div>

The above figure shows the convergence of the training and testing errors for the Adam Optimizer during the unit tests.

## References:

1) [RMSProp Optimizer - Tensorflow Implementation](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)

2) [Adam Optimizer - Tensorflow Implementation](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)