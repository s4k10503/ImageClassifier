from tkinter import filedialog
from injector import Injector, inject, singleton
from gui import ImageClassifierView
from inference import ImageClassifierModel


class ImageClassifierPresenter:
    """
    Presenter class in the MVP architecture for an image classifier app.

    It handles the interactions between the view and the model.
    """

    @inject
    def __init__(self, view: ImageClassifierView, model: ImageClassifierModel):
        """
        Initialize the presenter.

        Args:
            view (ImageClassifierView): The view part of the MVP.
            model (ImageClassifierModel): The model part of the MVP.
        """
        self.view = view
        self.model = model
        view.image_selected_subject.subscribe(
            on_next=lambda _: self.on_image_selected()
        )

    def setup_model(self):
        """
        Set up the image classification model with the necessary files.
        """
        model_path = "../model/resnet18-v1-7.onnx"
        labels_path = "../json/imagenet_class_index.json"
        self.model.load_model(model_path, labels_path)

    def start(self):
        """
        Start the GUI event loop.
        """
        self.view.mainloop()

    def on_image_selected(self):
        """
        Handle the image selection event.

        Opens a file dialog for the user to select an image and
        then displays and classifies the selected image.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.view.display_image(file_path)
            self.classify_image(file_path)

    def classify_image(self, image_path):
        """
        Perform image classification on the selected image.

        Args:
            image_path (str): The path to the selected image.
        """
        predicted_label, confidence = self.model.perform_inference(image_path)
        self.view.display_result(predicted_label, confidence)


class Dependency:
    """
    A class for managing dependencies in the application using the Injector.

    It configures how different components of the application are instantiated and managed.
    """

    def __init__(self) -> None:
        """
        Initialize the Dependency class with an Injector.
        """
        self.injector = Injector(self.configure)

    @staticmethod
    def configure(binder):
        """
        Configure the binder for dependency injection.

        Args:
            binder: The binder instance used in dependency injection.
        """
        binder.bind(ImageClassifierView,
                    to=ImageClassifierView, scope=singleton)
        binder.bind(ImageClassifierModel,
                    to=ImageClassifierModel, scope=singleton)
        binder.bind(ImageClassifierPresenter,
                    to=ImageClassifierPresenter, scope=singleton)


if __name__ == "__main__":
    dependency = Dependency()
    presenter = dependency.injector.get(ImageClassifierPresenter)
    presenter.setup_model()
    presenter.start()
