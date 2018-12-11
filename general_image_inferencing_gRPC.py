import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import label_map_util

# The input/output spec for the model is described in the Jupyter Notebook
#
# To serve the model, do something like:
#
#    sudo docker run -it -p 8501:8501 --mount type=bind,source=<path to model>,target=/models/<model name>\
#        -e MODEL_NAME=<model name> -t tensorflow/serving


def sem_segment(server_address, model_spec_name, image, label_map_path=None):
    """ Run a semantic segmentation inference on a TensorFlow Serving server.

    :param server_address: Address (hostname & port) of the TensorFlow Serving server.
    :param model_spec_name: The name of the model which you'd like the inference from.
    :param image: A PIL/Pillow image to inference on.
    :param label_map_path: The path to a label_map file, to be used to convert class ids to human-readable names.
    :return: A tuple of the lists of classes, labels, scores, boxes, and masks. Classes are integer ids, labels are
    text (or the entire list is `None` if no `label_map_path` was not specified), scores are floats on [0,1), boxes are
    lists of length 4 with bounds represented on [0,1]. The exact format of the masks isn't well-specified yet.
    """
    numpy_ary = np.array([np.array(image, dtype=np.uint8)])

    stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc.insecure_channel(server_address))
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(numpy_ary))

    result = stub.Predict(request, 10.0)
    outputs = result.outputs

    if label_map_path:
        label_map = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    else:
        label_map = None

    num_detections = int(outputs['num_detections'].float_val[0])
    classes = [int(x) for x in outputs['detection_classes'].float_val[:num_detections]]  # TODO: Finish conversion
    labels = [label_map[x]['name'] for x in classes] if label_map else None
    scores = outputs['detection_scores'].float_val[:num_detections]
    boxes = outputs['detection_boxes'].float_val[:num_detections] if outputs['detection_boxes'] else None
    masks = outputs['detection_masks'].float_val[:num_detections] if outputs['detection_masks'] else None

    return classes, labels, scores, boxes, masks


def obj_detect(server_address, model_spec_name, image, label_map_path=None):
    """ Run an object detection inference on a TensorFlow Serving server.

    :param server_address: Address (hostname & port) of the TensorFlow Serving server.
    :param model_spec_name: The name of the model which you'd like the inference from.
    :param image: A PIL/Pillow image to inference on.
    :param label_map_path: The path to a label_map file, to be used to convert class ids to human-readable names.
    :return: A tuple of the lists of classes, labels, scores, and boxes. Classes are integer ids, labels are text (or
    the entire list is `None` if no `label_map_path` was not specified), scores are floats on [0,1), boxes are lists of
    length 4 with bounds represented on [0,1].
    """
    classes, labels, scores, boxes, _ = sem_segment(server_address, model_spec_name, image, label_map_path)
    return classes, labels, scores, boxes


def classify(server_address, model_spec_name, image, label_map_path=None):
    """ Run a classification inference on a TensorFlow Serving server.

    :param server_address: Address (hostname & port) of the TensorFlow Serving server.
    :param model_spec_name: The name of the model which you'd like the inference from.
    :param image: A PIL/Pillow image to inference on.
    :param label_map_path: The path to a label_map file, to be used to convert class ids to human-readable names.
    :return: A tuple of the lists of classes, labels, and scores. Classes are integer ids, labels are text (or the
    entire list is `None` if no `label_map_path` was not specified), scores are floats on [0,1).
    """
    classes, labels, scores, _, _ = sem_segment(server_address, model_spec_name, image, label_map_path)
    return classes, labels, scores
