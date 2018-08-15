---
layout: post
title: Adagrad and Adadelta Optimizers - Implementation and Testing:-
---

In this blog post, I'll be explaining the implementation of Adagrad and Adadelta optimizers.

## Adagrad:

AdaGrad is an optimization method that allows different step sizes for different features. It increases the influence of rare but informative features i.e. It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing with sparse data.

Thus, the update is implemented as follows, ( similar to the tensorflow implementation )

```
Vt = Vt-1 + currentSquaredGradients
theta = theta - learningRate * currentGradients / (sqrt(Vt + epsilon))
```

So, one step of update is performed as,

$$
\begin{align}
\begin{split}
v_t &= v_{t-1} + \nabla_\theta^2 J( \theta) \\  
\theta &= \theta - \dfrac{\eta}{\sqrt{v_t + \epsilon}} \nabla_\theta J( \theta)
\end{split}
\end{align}
$$

## Testing Adagrad:

I used the same unit tests approach as for SGD optimizer. Have a look at **Testing the SGD optimizer post**.

<div>
    <a href="https://plot.ly/~ravikiran0606/33/?share_key=52ETON5ZthCr9zXsqp3XN6" target="_blank" title="ADAGRADUTP" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/33.png?share_key=52ETON5ZthCr9zXsqp3XN6" alt="ADAGRADUTP" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:33" sharekey-plotly="52ETON5ZthCr9zXsqp3XN6" src="https://plot.ly/embed.js" async></script>
</div>

The above figure shows the convergence of the training and testing errors for the Adagrad Optimizer during the unit tests.

## Adadelta:

Adadelta is an extension of Adagrad that tries to reduce the monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta stores only the window of accumulated past gradients to some fixed window size $$w$$.

Thus, the update is implemented as follows, ( similar to the tensorflow implementation )

```
Vt = rho * Vt-1 + (1-rho) * currentSquaredGradients
currentUpdates = sqrt(Wt + epsilon) * currentGradients / sqrt(Vt + epsilon)
theta = theta - learningRate * currentUpdates
Wt = rho * Wt-1 + (1-rho) * currentSquaredUpdates

```

So, one step of update is performed as,

$$
\begin{align}
\begin{split}
v_t &= \rho v_{t-1} + (1-\rho) \nabla_\theta^2 J( \theta) \\ 
\Delta\theta &= \dfrac{\sqrt{w_t + \epsilon}}{\sqrt{v_t + \epsilon}} \nabla_\theta J( \theta) \\
\theta &= \theta - \eta \Delta\theta \\ 
w_t &= \rho w_{t-1} + (1-\rho) \Delta\theta^2
\end{split}
\end{align}
$$

## Testing Adadelta:

I used the same unit tests approach as for SGD optimizer. Have a look at **Testing the SGD optimizer post**.

<div>
    <a href="https://plot.ly/~ravikiran0606/34/?share_key=48EXVp2d3ovHjvc3irP914" target="_blank" title="ADADELTAUTP" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/34.png?share_key=48EXVp2d3ovHjvc3irP914" alt="ADADELTAUTP" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:34" sharekey-plotly="48EXVp2d3ovHjvc3irP914" src="https://plot.ly/embed.js" async></script>
</div>

The above figure shows the convergence of the training and testing errors for the Adadelta Optimizer during the unit tests.

## References:

1) [Adagrad Optimizer - Tensorflow Implementation](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

2) [Adadelta Optimizer - Tensorflow Implementation](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer)