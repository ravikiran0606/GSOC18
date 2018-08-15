---
layout: post
title: Comparison of various optimizers and future work:-
---

In this blog post, I will be comparing all the optimizers on the [same dataset](http://root.cern.ch/files/tmva_class_example.root) that is used for performing classification using TMVA.

<div>
    <a href="https://plot.ly/~ravikiran0606/45/?share_key=nJ69a8LTr8AdxGq8LxozR3" target="_blank" title="TMVA Optimizers - Training Errors" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/45.png?share_key=nJ69a8LTr8AdxGq8LxozR3" alt="TMVA Optimizers - Training Errors" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:45" sharekey-plotly="nJ69a8LTr8AdxGq8LxozR3" src="https://plot.ly/embed.js" async></script>
</div>

<div>
    <a href="https://plot.ly/~ravikiran0606/47/?share_key=4Z6uv7ZaWyQgzsoHun8ETv" target="_blank" title="TMVA Optimizers - Test Errors" style="display: block; text-align: center;"><img src="https://plot.ly/~ravikiran0606/47.png?share_key=4Z6uv7ZaWyQgzsoHun8ETv" alt="TMVA Optimizers - Test Errors" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="ravikiran0606:47" sharekey-plotly="4Z6uv7ZaWyQgzsoHun8ETv" src="https://plot.ly/embed.js" async></script>
</div>

The above figures show the convergence of the training and testing erros of various optimizers during the integration tests ( methodDL tests ).

## Future Work:

1) Implement other optimizers like **Adamax, Nadam and Nesterov accelerated SGD** optimizers.<br>
2) Add **Weight Decay of learning rate** implementation to optimizers.<br>
3) **Benchmark the individual optimizers** on separate datasets with tensorflow.
