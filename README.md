# gpssi
This repository provides code for the paper:

_Lê, Matthieu, et al. "Sampling image segmentations for uncertainty quantification." Medical image analysis 34 (2016): 42-51. doi:10.1016/j.media.2016.04.005_


Note that this is not an official implementation by the authors and was motivated by the lack of publicly available code. 
Be aware that there might be difference to the original implementation.


## Implementation Details

### Geodesic map

To produce the geodesic maps, this project relies on the _GeodisTK_ packages. This package can be installed via pip or via 
source code ([https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)). Due to observed issues when installing 
the package via pip, we suggest to install it from the github link (see [installation](#installation)).


### Factorization via Kronecker
The authors use a Kronecker matrix representation of the covariance matrix to overcome the issue of large covariance matrices.

This project implements the kronecker matrix-vector product based on following reference:
- _Saatçi, Yunus. Scalable inference for structured Gaussian process models. Diss. University of Cambridge, 2012._
- _Gilboa, Elad, Yunus Saatçi, and John P. Cunningham. "Scaling multidimensional inference for structured Gaussian processes." IEEE transactions on pattern analysis and machine intelligence 37.2 (2013): 424-436._


## Installation
This projects is available as python package and can be installed by 

```pip install gpssi```

Otherwise, the package can also from the source code via 

```
git clone https://github.com/alainjungo/gpssi.git
cd gpssi
pip install .
```

The _GeodisTK_ package is not installed automatically and has to be installed manually. 
We propose to use the direct installation via source code: 

```
pip install git+ssh://git@github.com/taigw/GeodisTK.git
```

Alternatively, you can try installing it from pypi (`pip install GeodisTK`)


## Usage
See [gpssi_example.py](examples/gpssi_example.py) for an example usage.

