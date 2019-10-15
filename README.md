# Face Alignment using a Deeply-initialized Coarse-to-fine Ensemble of Regression Trees

We provide C++ code in order to replicate the face alignment experiments in our paper
http://openaccess.thecvf.com/content_ECCV_2018/papers/Roberto_Valle_A_Deeply-initialized_Coarse-to-fine_ECCV_2018_paper.pdf

If you use this code for your own research, you must reference our conference and journal papers:

```
A Deeply-initialized Coarse-to-fine Ensemble of Regression Trees for Face Alignment
Roberto Valle, José M. Buenaposada, Antonio Valdés, Luis Baumela.
European Conference on Computer Vision, ECCV 2018, Munich, Germany, September 8-14, 2018.
```

```
Face Alignment using a 3D Deeply-initialized Ensemble of Regression Trees
Roberto Valle, José M. Buenaposada, Antonio Valdés, Luis Baumela.
Computer Vision and Image Understanding, CVIU 2019.
https://doi.org/10.1016/j.cviu.2019.102846
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework
- ert_simple https://github.com/bobetocalo/ert_simple

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── alignment
        └── ert_simple
        └── bobetocalo_eccv18
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --database option to load the proper trained model.
```
> ./release/face_alignment_bobetocalo_eccv18_test --database 300w_public --measure pupils
```
