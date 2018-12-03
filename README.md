# General Image Inferencing

This is a demonstration of a generalized framework to get inferences from classification, object detection, and semantic
segmentation models using TensorFlow Serving. This work is heavily based on [the object_detection section of the TensorFlow
Models repository](https://github.com/tensorflow/models/tree/master/research/object_detection).

## This Repository

- `general_image_inferencing.py`: This is the main demonstration.
- `general_image_inferencing.ipynb`: This is the original demo, which uses gRPC. It seems like gRPC is the more efficient
  option, but it's poorly documented, and *difficult* in some ways. So even though I got it to work with gRPC here, I
  decided to switch to the REST API with JSON in general.
- Several files lifted from [the object_detection section of the TensorFlow Models repository](
  https://github.com/tensorflow/models/tree/master/research/object_detection/utils/label_map_util.py):
  - `label_map_util.py`: This contains utilities for reading the text-format protobufs which contain label maps.
  - `data/`: Some sample label maps.
  - `string_int_label_map.proto`: The prototype for the label maps.
- `string_int_label_map_pb2.py`: The result of compiling `string_int_label_map.proto`.
- `brompton.jpg`: A sample image. Public domain. https://commons.wikimedia.org/wiki/File:2_red_Bromptons_10Av_30_jeh.JPG

## Model Specification

This notebook is a demonstration of a single system which could be used for inferencing on any number of classification,
object detection, or semantic segmentation models. To make this possible, the model being served has to conform to a
specification.

### Input

let `n_img` := 1. TODO: Allow processing of more than one image at a time?

- `inputs`, type: `uint8`, shape: `[n_img, -1, -1, 3]`. This is the image. 

### Output

- `num_detections`, type: `int`, shape: `[n_img]`. The value of this field is always less than or equal to `len_detections`.
- `detection_classes`, type: `int`, shape: `[n_img, len_detections]`. Class identifier.
- `detection_scores`, type: `float`, shape: `[n_img, len_detections]`. Confidence level.
- `detection_boxes`, type: `float`, shape: `[n_img, len_detections, 4]`. Values are on `[0,1)`, representing the bounds of
  the boxes. (object detection & semantic segmentation only)
- `detection_masks`, type: `int`, shape: `[n_img, len_detections, -1, -1]`. Value is either `1` or `0`, meaning to be
  described. (semantic segmentation only)

## Serving the model

Do something like:

    sudo docker run -it -p 8501:8501\
        --mount type=bind,source=<path to model>,target=/models/<model name>\
        -e MODEL_NAME=<model name>\
        -t tensorflow/serving
