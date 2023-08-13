from gui import ImageClassifierApp
from inference import perform_inference, load_labels, initialize_onnx_model

from PIL import Image, ImageTk
from tkinter import filedialog
import warnings


def select_image():
    """
    Callback function to select an image using a file dialog.

    Args:
        None

    Returns:
        None
    """
    global tk_image
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)

    max_thumbnail_size = (256, 256)
    image.thumbnail(max_thumbnail_size)
    tk_image = ImageTk.PhotoImage(image)
    app.label_image.configure(image=tk_image)

    predicted_label, confidence = perform_inference(file_path, sess, input_name, class_labels)
    app.label_result.configure(text="Predicted Label: " + predicted_label)
    app.label_confidence.configure(text=f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":

    warnings.simplefilter('ignore')

    # Path to the JSON file containing class labels for the model
    json_path = '../json/imagenet_class_index.json'

    # Path to the ONNX model file
    model_path = '../model/resnet18-v1-7.onnx'

    # Load class labels from JSON file
    class_labels = load_labels(json_path)

    # Initialize the ONNX model session and get the input name
    sess, input_name = initialize_onnx_model(model_path)
    tk_image = None

    # Create an instance of the ImageClassifierApp class
    app = ImageClassifierApp(select_image)

    # Start the main event loop of the GUI application
    app.mainloop()
