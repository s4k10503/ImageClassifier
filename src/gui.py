import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from rx.subject.subject import Subject

FONT_TYPE = "Arial"


class ImageClassifierView(ctk.CTk):
    """
    A custom Tkinter class for creating the GUI of an image classifier app.
    This class manages the layout and UI elements for image classification.
    """

    def __init__(self):
        """
        Initializes the ImageClassifierView object.
        Sets up the initial UI components and state.
        """
        super().__init__()
        self.tk_image = None
        self.image_selected_subject = Subject()
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the user interface for the ImageClassifierView.

        This includes setting the appearance mode, color theme, form size,
        and initializing the main frame and its widgets.
        """
        # Set CustomTkinter form design settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set form size
        self.geometry("500x500")
        self.title("Image Classifier")

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.create_widgets(main_frame)

    def create_widgets(self, main_frame):
        """
        Create and add widgets to the main frame.

        Args:
            main_frame: The main frame in which widgets are placed.
        """
        # "Select Image" button
        self.button_select = ctk.CTkButton(
            main_frame, text="Select Image",
            command=self.on_select_image, font=(FONT_TYPE, 20))
        self.button_select.pack(pady=10)

        # Image label
        self.label_image = ctk.CTkLabel(
            main_frame, text="", width=300, height=300)
        self.label_image.pack(pady=10)

        # Predicted label
        self.label_result = ctk.CTkLabel(
            main_frame, text="Predicted Label: ", font=(FONT_TYPE, 20))
        self.label_result.pack(pady=5)

        # Confidence level
        self.label_confidence = ctk.CTkLabel(
            main_frame, text="Confidence: ", font=(FONT_TYPE, 20))
        self.label_confidence.pack(pady=5)

    def on_select_image(self):
        """
        Handle the event when the 'Select Image' button is clicked.

        Currently, it triggers an event in the image_selected_subject.
        """
        self.image_selected_subject.on_next(None)

    def display_image(self, image_path):
        """
        Display an image on the UI.

        Args:
            image_path: The path of the image to be displayed.
        """
        image = Image.open(image_path)
        max_thumbnail_size = (256, 256)
        image.thumbnail(max_thumbnail_size)
        tk_image = ImageTk.PhotoImage(image)
        self.label_image.configure(image=tk_image)
        self.tk_image = tk_image  # Keep a reference

    def display_result(self, result, confidence):
        """
        Display the classification result and confidence on the UI.

        Args:
            result: The classification result.
            confidence: The confidence level of the classification.
        """
        self.label_result.configure(text="Predicted Label: " + result)
        self.label_confidence.configure(text=f"Confidence: {confidence:.2f}%")
