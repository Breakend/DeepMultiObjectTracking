Caltech Pedestrian Dataset Converter
============================

# Requirements

- OpenCV 3.0+ (with Python binding)
- Python 2.7+, 3.4+, 3.5+
- NumPy 1.10+
- SciPy 0.16+

# Caltech Pedestrian Dataset

```
$ bash shells/download.sh
$ python scripts/convert_annotations.py
$ python scripts/convert_seqs.py
```

Each `.seq` movie is separated into `.png` images. Each image's filename is consisted of `{set**}_{V***}_{frame_num}.png`. According to [the official site](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), `set06`~`set10` are for test dataset, while the rest are for training dataset.  
Annotations will be output as COCO format .xml file (1 for each image).

# Output
Example output: Inside target directory: ```target_directory/set00/V006/*.jpg```

```
$ python tests/test_plot_annotations.py
```
