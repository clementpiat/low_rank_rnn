<p align="center"><b><i>
	Low-rank RNN
</b></i></p>

<div align="center">

![Human Written](https://img.shields.io/badge/code-human_written-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

</div>


## Go - NoGo

I generated synthetic stimuli data for simulating a Go-NoGo task and trained a rank-two RNN on it.

```shell
pip install -r requirements.txt
python -m low_rank_rnn.train
```

&rarr; We see that the neurons activity lies on a 1D-manifold:

### Go
![go](./low_rank_rnn/figures/go.gif)
### NoGo
![nogo](./low_rank_rnn/figures/nogo.gif)