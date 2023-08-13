import tkinter as tk
import customtkinter as ctk

FONT_TYPE = "Arial"


class ImageClassifierApp(ctk.CTk):
    """
    A GUI application for image classification using CustomTkinter.

    Args:
        select_image_callback (callable): Callback function to handle image selection.

    Attributes:
        button_select (customtkinter.CTkButton): Button to select an image.
        label_image (customtkinter.CTkLabel): Label to display the selected image.
        label_result (customtkinter.CTkLabel): Label to display the predicted label.
        label_confidence (customtkinter.CTkLabel): Label to display the confidence level.
    """

    def __init__(self, select_image_callback):
        """
        Initialize the ImageClassifierApp.

        Args:
            select_image_callback (callable): Callback function to handle image selection.
        """
        super().__init__()

        # Set CustomTkinter form design settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set form size
        self.geometry("500x500")
        self.title("Image Classifier")

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Display the "Select Image" button
        self.button_select = ctk.CTkButton(main_frame, text="Select Image", command=select_image_callback, font=(FONT_TYPE, 20))
        self.button_select.pack(pady=10)

        # Display the image label
        self.label_image = ctk.CTkLabel(main_frame, text="", width=100, height=100)
        self.label_image.pack(pady=10)

        # Display the predicted label
        self.label_result = ctk.CTkLabel(main_frame, text="Predicted Label: ", font=(FONT_TYPE, 20))
        self.label_result.pack(pady=5)

        # Display the confidence level
        self.label_confidence = ctk.CTkLabel(main_frame, text="Confidence: ", font=(FONT_TYPE, 20))
        self.label_confidence.pack(pady=5)
