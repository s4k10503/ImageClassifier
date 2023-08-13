# Image Classifier

## Overview

The Image Classifier project is a user-friendly application that allows users to classify images using a pre-trained ONNX model. The application provides a graphical user interface (GUI) built with CustomTKInter, allowing users to select an image, display its thumbnail, and perform classification to predict the label and confidence level.

## Features

Image Selection: Enables users to select an image using a file dialog.
Thumbnail Display: Resizes and displays the selected image as a thumbnail.
Inference: Performs inference on the selected image and displays the predicted label and confidence percentage.

## Dependencies

- PIL (Pillow) for image processing
- CustomTKInter for customizing the GUI
- Tkinter for the basic GUI framework
- ONNX Runtime for handling ONNX models
- NumPy for numerical operations
- Requests for HTTP requests (in setup)
- Standard libraries: os, json, urllib.request, shutil, warnings

## Setup

Please run the setup.py to download the required files.

## Usage

Run main.py to launch the application, and use the GUI to select an image for classification.

## Contributing

Feel free to fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project utilizes the ResNet ONNX model provided by ONNX Model Zoo, which is licensed under the Apache License, Version 2.0. You must comply with the terms of that license when using the model.
The code and other components of this project are licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
