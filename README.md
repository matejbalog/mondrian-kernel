# The Mondrian Kernel

Matej Balog, Balaji Lakshminarayanan, Zoubin Ghahramani, Daniel M. Roy, Yee Whye Teh

*Thirty-Second Conference on Uncertainty in Artificial Intelligence (UAI), 2016.*

[[PDF](http://www.auai.org/uai2016/proceedings/papers/236.pdf)] 
[[supp](http://www.auai.org/uai2016/proceedings/supp/236_supp.pdf)] 
[[arXiv](https://arxiv.org/abs/1606.05241)] 
[[poster](http://matejbalog.eu/research/mondrian_kernel_poster.pdf)]
[[slides](http://matejbalog.eu/research/mondrian_kernel_slides.pdf)]

The scripts provided here implement experiments from this paper. The scripts `experiment_1_laplace_kernel_approximation`, `experiment_2_fast_kernel_width_learning` and `experiment_3_mondrian_kernel_vs_forest` are intended to be directly runnable.

### Requirements

Python packages: `heapq`, `matplotlib`, `numpy`, `scipy`, `sklearn`, `sys`, `time`

The CPU dataset `cpu.mat` can be download and extracted from [here](https://keysduplicated.com/~ali/random-features/data/cpu.tgz).

### Known issues

A [bug](http://stackoverflow.com/questions/35283073/scipy-io-loadmat-doesnt-work) in `scipy` may cause the Python kernel to restart when loading the CPU dataset from `cpu.mat`. Downgrading to `scipy 0.16.0` should solve the problem. 

### BibTeX

```
@inproceedings{balog2016mondriankernel,
  author = {Matej Balog and Balaji Lakshminarayanan and Zoubin Ghahramani and Daniel M.~Roy and Yee Whye Teh},
  title={The {M}ondrian Kernel},
  booktitle = {32nd Conference on Uncertainty in Artificial Intelligence (UAI)},
  year = {2016},
  month = {June},
  url = {http://www.auai.org/uai2016/proceedings/papers/236.pdf}
}
```
