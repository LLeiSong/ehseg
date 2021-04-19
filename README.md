## ehseg
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

This is a python package to do edge-highlight image segmentation.

### Introduction

This package is an implement of Edge-highlight Image Segmentation (EHSEG) to do image segmentation.
It roughly has five steps and an optional step:
1. First uses mean-shift with iterations to filter the image.
2. Then calculate the average of edge gradient magnitudes generated by multiple methods. The methods include sobel, scharr, roberts,
    prewitt, laplace, and laplace gaussian.
3. Extract skeleton using adaptive threshold.
4. Highlight the image using the skeleton.
5. Run region_growing image segmentation from GRASS GIS to finish the segmentation.
6 (Optional) simplify the segmentation polygons using Douglas-Peucker Algorithm.

The method allows doing segmentation on one image, or a pair of images (e.g. growing-season image and off-season image).

### Usage

The function `ehseg` is the main function to do segmentation. Here is an example of how to use it:

The user must install GRASS GIS first, and provide `GRASSBIN` and `gisbase`.

```
img_paths = ['planet_medres_normalized_analytic_2017-12_2018-05_mosaic_1219-998.tif',
             'planet_medres_normalized_analytic_2018-06_2018-11_mosaic_1219-998.tif']
dst_path = '/Users/leisong/downloads'
opt_name = 'segments_1219-998'
bands = [1, 2, 3, 4]
grassbin = '/Applications/GRASS-7.8.app/Contents/MacOS/Grass.sh'  # for Mac
gisbase = '/Applications/GRASS-7.8.app/Contents/Resources'  # for Mac
n_iter = 3
max_window = 5
window_size_thred = 101
method = 'separate'
ram = 16
threshold = 0.2
similarity = "manhattan"
minsize = 10
iterations = 200
vectorize = True
simplify_thred = 3.0
keep = False

ehseg(img_paths=img_paths,
      dst_path=dst_path,
      opt_name=opt_name,
      bands=bands,
      grassbin=grassbin,  # for Mac
      gisbase=gisbase,  # for Mac
      n_iter=n_iter, max_window=max_window,
      window_size_thred=window_size_thred,
      method=method,
      ram=ram,
      threshold=threshold,
      similarity=similarity,
      minsize=minsize,
      iterations=iterations,
      vectorize=vectorize,
      simplify_thred=simplify_thred,
      keep=keep)
```

You could check more details about the arguments from function help documentation.

### Acknowledgement

This package is part of project ["Combining Spatially-explicit Simulation of Animal Movement and Earth Observation to Reconcile Agriculture and Wildlife Conservation"](https://github.com/users/LLeiSong/projects/2).
This project is funded by NASA FINESST program (award number: 80NSSC20K1640).
