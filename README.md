## padded-transformations
Providing padded versions of OpenCV's `warpAffine()` and `warpPerspective()` functions.

![Example image](example.png "Example output")

## usage

```python
import padtransf
src_warped, dst_padded = padtransf.warpPerspectivePadded(src, dst, homography)
src_warped, dst_padded = padtransf.warpAffinePadded(src, dst, affine_transf)
```

## sources

Read [my Stack Overflow answer](https://stackoverflow.com/questions/44457064/displaying-stitched-images-together-without-cutoff-using-warpaffine/44459869#44459869) which inspired this repository.

The images used to produce `example.png` are from [Oxford's Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/data/affine/).
