# gpssi
This repository provides code for the paper:

_Lê, Matthieu, et al. "Sampling image segmentations for uncertainty quantification." Medical image analysis 34 (2016): 42-51. doi:10.1016/j.media.2016.04.005_


Note that this is not an official implementation by the authors and was motivated by the lack of publicly available code. 
Be aware that there might be difference to the original implementation.


## Implementation Details

### Geodesic map

To produce the geodesic maps, this project relies on the _GeodisTK_ packages. This package can be installed via pip.

`pip install GeodisTK`
 
However, you might need to clone and setup it manually when the pip installation fails or the execution 
doesn't behave as desired.

```
git clone https://github.com/taigw/GeodisTK
pip install -e <path_to_GeodisTK>
``` 

### Factorization via Kronecker
The authors use a Kronecker matrix representation of the covariance matrix to overcome the issue of large covariance matrices.

This project implements the kronecker matrix-vector product based on following reference:
- _Saatçi, Yunus. Scalable inference for structured Gaussian process models. Diss. University of Cambridge, 2012._
- _Gilboa, Elad, Yunus Saatçi, and John P. Cunningham. "Scaling multidimensional inference for structured Gaussian processes." IEEE transactions on pattern analysis and machine intelligence 37.2 (2013): 424-436._
 

## Installation
You can install the required package by

`pip install -r requirements.txt`

The _GeodisTK_ package is not listed in the requirements file. You will need to install it manually.

## Usage
See [gpssi_example.py](examples/gpssi_example.py) for an example usage.

