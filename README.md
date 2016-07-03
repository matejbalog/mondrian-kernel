# The Mondrian Kernel

Matej Balog, Balaji Lakshminarayanan, Zoubin Ghahramani, Daniel M. Roy, Yee Whye Teh

*Thirty-Second Conference on Uncertainty in Artificial Intelligence (UAI), 2016.*

[[PDF](http://www.auai.org/uai2016/proceedings/papers/236.pdf)] 
[[supp](http://www.auai.org/uai2016/proceedings/supp/236_supp.pdf)] 
[[arXiv](https://arxiv.org/abs/1606.05241)]

The scripts provided here implement experiments from this paper. The scripts `experiment_1_laplace_kernel_approximation`, `experiment_2_fast_kernel_width_learning` and `experiment_3_mondrian_kernel_vs_forest` are intended to be directly runnable.

### Requirements

Python packages: `heapq`, `matplotlib`, `numpy`, `scipy`, `sklearn`, `sys`, `time`

The CPU dataset `cpu.mat` can be download and extracted from [here](https://keysduplicated.com/~ali/random-features/data/cpu.tgz).

### Known issues

A [bug](http://stackoverflow.com/questions/35283073/scipy-io-loadmat-doesnt-work) in `scipy` may cause the Python kernel to restart when loading the CPU dataset from `cpu.mat`. Downgrading to `scipy 0.16.0` should solve the problem. 

### BibTeX

``@article{balog2016mondriankernel,
  title={The Mondrian Kernel},
  author={Balog, Matej and Lakshminarayanan, Balaji and Ghahramani, Zoubin and Roy, Daniel M and Teh, Yee Whye},
  journal={Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence (UAI-16), Jersey City},
  year={2016}
}``
