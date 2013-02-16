Dataset name:
US Postal Service Handwritten Digits

Internet Source of dataset:
http://www.gaussianprocess.org/gpml/data/

Additional preprocessing:
Scaling the data from 0 to 1.
Saving as ASCII data for programming languages other than MATLAB,
so double precision is lost, but this will not matter for this dataset.

Dataset description (from http://www.gaussianprocess.org/gpml/data/):
The data has traditionally been used in a splitting of 7291 cases for training and 2007 cases for testing. However, these two sets were actually collected in slightly different ways, and thus not very suitable for demonstrating learning algorithms. (In fact, it is pretty well know by machine learning practitioners with experience on this data, that the cases in the test set are much harder to classify that the cases in the training set.) To avoid these problems, we concatenated both sets, randomly reshuffeled the cases, and divided them anew into training and test sets, containing 4649 cases each (this also avoid the problem with the previous partitions having too few test cases to be able to say very much about the statistical significance of the difference between the performance of different algorithms). The file contains 4 variables:

  test_labels         10x4649                371920  double array
  test_patterns      256x4649               9521152  double array
  train_labels        10x4649                371920  double array
  train_patterns     256x4649               9521152  double array

The *_patterns variables contain a raster scan of the 16 by 16 grey level pixel intensities, which have scaled such that the range is [-1; 1]. The *_labels variables contain a one-of-k encoding with values -1 and +1 of the classification; one +1 per column.

The USPS digits data were gathered at the Center of Excellence in Document Analysis and Recognition (CEDAR) at SUNY Buffalo, as part of a project sponsored by the US Postal Service. The dataset is described in A Database for Handwritten Text Recognition Research, J. J. Hull, IEEE PAMI 16(5) 550-554, 1994.