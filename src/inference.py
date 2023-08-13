import onnxruntime as ort
import numpy as np
from PIL import Image
import json


def load_labels(json_path):
    """
    Load class labels from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing class labels.

    Returns:
        dict: A dictionary mapping class index to class label.
    """

    with open(json_path, 'r') as file:
        labels = json.load(file)

    return {int(idx): label[1] for idx, label in labels.items()}


def preprocess_image(image_path):
    """
    Preprocess an image for inference.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """

    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32')

    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')

    return (image_array / 255 - mean) / std


def softmax(x):
    """
    Compute the softmax of an array of values.

    Args:
        x (numpy.ndarray): Input array of values.

    Returns:
        numpy.ndarray: Softmax probabilities.
    """

    exp_x = np.exp(x - np.max(x))

    return exp_x / exp_x.sum(axis=1, keepdims=True)


def perform_inference(image_path, sess, input_name, class_labels):
    """
    Perform inference on an image using an ONNX model.

    Args:
        image_path (str): Path to the image file.
        sess (onnxruntime.InferenceSession): ONNX model inference session.
        input_name (str): Name of the input tensor.
        class_labels (dict): Dictionary mapping class index to class label.

    Returns:
        tuple: Predicted label and confidence as a tuple.
    """

    input_data = preprocess_image(image_path).transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    result = sess.run(None, {input_name: input_data})
    result_probabilities = softmax(result[0])
    predicted_index = np.argmax(result_probabilities)

    predicted_label = class_labels[predicted_index]
    confidence = result_probabilities[0][predicted_index] * 100

    return predicted_label, confidence


def initialize_onnx_model(model_path):
    """
    Initialize an ONNX model for inference.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        tuple: Inference session and input tensor name.
    """

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    return sess, input_name
