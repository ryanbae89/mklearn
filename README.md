# Mykit-Learn

My implementation of scikit-learn, mykit-learn. Mklearn is my implementation of Linear Support Vector Machine model. It is implemented similarly to scikit-learn using classes and fit functions, etc. 

## Usage

Make sure Python has the correct path to the modules, and simply import as follows:

```python
import mklearn
import multiclass

mklearn.myLinearSVC(...)
multiclass.Multiclass(...)
```
## Implementation

### Linear Support Vector Classifier

`mklearn.py` contains the implementation of linear support vector classifier. The instance of class `myLinearSVC` can fit a single binary classifier model using linear SVC. The class is structured in a similar way to scikit-learn's convention, where the class is declared with hyperparameters and `.fit()` method is called to train the model.

### Multiclass Classification

Multiclass classification is supported by `multiclass.py` file. It contains `Multiclass` object and several prediction functions that wrap around `myLinearSVC`. `Multiclass` can perform both one-vs-one and one-vs-rest multiclass classification through the argument `multiclass`.

The module also supports model specific cross-validation while running multiclass classification. Simply pass in a list of regularization penalties as `Cs` and set `k` to the number of desired folds for cross-validation.

Finally, `Multiclass` class supports limited multiprocessing to speed up training. Currently, it is only available for the prediction portion only. You can pass in number of threads into `n_threads` argument. The multiprocessing capabilities for training portion is in development.

## Dependencies

The following dependencies are needed to install and run mklearn. 

`numpy, pandas, matplotlib, scipy, sklearn`

Currently mkleran is only supported in Python 3. 

## Tutorial

There are 3 tutorials to showcase how to use mklearn. 

* `mklearn_tutorial_real_dataset.ipynb`: contains mklearn multiclass example on real-world dataset. 

* `mklearn_tutorial_simulated_dataset.ipynb`: contains mklearn multiclass example on simulated dataset using sklearn.

* `tutorial_sklearn_comparison.ipynb`: contains performance comparisons with similar implementations from sklearn.

Take a look at the mentioned notebooks for details.

## Support/Contact

Author: Ryan Bae

Initial Commit Date: 6/02/2018

This is my official code release for DATA 558 Spring 2018 at University of Washington's MS Data Science program.



