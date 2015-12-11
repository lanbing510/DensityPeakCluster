# DensityPeakCluster

Python codes for 'Clustering by fast search and find of density peaks' in science 2014.  

## Introduction

I forked the original DensityPeakCluster from [here](https://github.com/jasonwbw/DensityPeakCluster), thanks jasonwbw. I have fixed its bugs and reproduced the excellent work of Alex Rodriguez and Alessandro Laio in the paper 'Clustering by fast search and find of density peaks'. The matlab code of Alex Rodriguez and Alessandro Laio was pleaced under ./data/data_in_paper. The results of their code are shown below:


The results of our code are shown here:


Note that the mds method of python is not totally same with mds in matlab. However, we still can see the results are the same in essence.

## How to Use
  
Step0: If your data is not the distance between points but the points' vector, write you distance builder in distance like distance_builder_data_iris_flower.py.  
Step1: Change the data file in step1_choose_center.py, then run it to choose cluster threshold.  
Step2: Change the data file and threshold in step2_cluster.py, then run it.  
```python
python distance_builder_data_iris_flower.py
python step1_choose_center.py
python step2_cluster.py
```

## Dependencies
- [NumPy](http://www.numpy.org): normal computing
- [Matplotlib](http://matplotlib.sourceforge.net/): For plotting data to choose threshold
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn): use for mds to plot result

## Reference
- [Clustering by fast search and find of density peaks](http://www.sciencemag.org/content/344/6191/1492.full)

## License
The MIT License (MIT)
