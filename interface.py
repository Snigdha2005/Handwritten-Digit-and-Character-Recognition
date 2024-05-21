import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from Preprocessing.data_prep import le  
from CNN.cnn import cnn_model


class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")

        # Initialize variables
        self.selected_option = tk.StringVar()  # To track the selected option
        self.img_array = None  # To store the image array from canvas drawing
        self.strokes = []

        self.create_widgets()

    def create_widgets(self):
        # Frame for options
        self.option_frame = tk.Frame(self.root)
        self.option_frame.pack(pady=10)

        #Frame to hold canvas and result display
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Radio buttons for options
        self.upload_radio = tk.Radiobutton(self.option_frame, text="Upload Image", variable=self.selected_option, value="upload", command=self.upload_image, font=("Algerian", 25))
        self.upload_radio.grid(row=0, column=0, padx=10)

        self.draw_radio = tk.Radiobutton(self.option_frame, text="Draw on Canvas", variable=self.selected_option, value="draw", font=("Algerian", 25))
        self.draw_radio.grid(row=0, column=1, padx=10)

        # Canvas for drawing
        self.canvas = tk.Canvas(self.main_frame, width=600, height=600, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Clear Canvas button
        self.clear_btn = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas, font=("Comic Sans MS", 18))
        self.clear_btn.pack(pady=10)

        # Predict button
        self.predict_btn = tk.Button(self.root, text="Predict", command=self.predict, font=("Comic Sans MS", 24))
        self.predict_btn.pack(pady=20)

        # Result display
        self.result_frame = tk.Frame(self.main_frame)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_label = tk.Label(self.result_frame, text="Prediction: ", font=("Comic Sans MS", 24))
        self.result_label.pack(pady=20)

        self.accuracy_label = tk.Label(self.result_frame, text="Accuracy: ", font=("Comic Sans MS", 24))
        self.accuracy_label.pack()

        # # Upload image button
        # self.upload_btn = tk.Button(self.root, text="Upload Image", command=self.upload_image, font=("Comic Sans MS", 24))
        # self.upload_btn.pack(pady=10)

        # Bind mouse events for canvas drawing
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

    def predict(self):
        # print("Selected option:", self.selected_option.get())  # Debug print
        # print("Image array:", self.img_array)  # Debug print
        # print("strokes first: ", self.strokes)
        if self.selected_option.get() == "upload" and self.img_array is not None:

            # # Convert image array back to image
            # img = Image.fromarray((self.img_array.reshape(28, 28) * 255).astype(np.uint8), mode='L')
            # # Display the image
            # img.show()

            prediction = cnn_model.predict(self.img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_class])[0]
            accuracy = prediction[0][predicted_class]
            self.result_label.config(text=f"Prediction: {predicted_label}")
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
        elif self.selected_option.get() == "draw":
            if self.strokes:
                # Convert the drawn strokes to an image array for prediction
                drawn_image = self.create_image_from_strokes()


                #UNCOMMENT THIS PART TO SEE WHAT IMAGE IS BEING GENERATED FROM THE IMAGE ARRAY FORMED FROM THOSE STROKES DRAWN BY US IN THE CANVAS PART
                # # Convert image array back to image
                # img = Image.fromarray((drawn_image.reshape(28, 28) * 255).astype(np.uint8), mode='L')
                # # Display the image
                # img.show()


                # Use the model to predict the handwritten digit or character
                prediction = cnn_model.predict(drawn_image)
                predicted_class = np.argmax(prediction)
                predicted_label = le.inverse_transform([predicted_class])[0]
                accuracy = prediction[0][predicted_class]

                # Update the result labels
                self.result_label.config(text=f"Prediction: {predicted_label}")
                self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
        else:
            messagebox.showerror("Error", "Please select an option and either upload an image or draw on the canvas.")

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        # Display the image on the canvas
        self.img = load_img(file_path, color_mode='grayscale', target_size=(28, 28))
        self.img_array = img_to_array(self.img).reshape(1, 28, 28, 1) / 255.0
        self.display_img = self.img.resize((500, 500))
        self.photo_img = ImageTk.PhotoImage(self.display_img)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x_center = (canvas_width) // 2
        y_center = (canvas_height) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=self.photo_img)

        # Enable predict button
        self.predict_btn.config(state=tk.NORMAL)

    def draw_on_canvas(self, event):
        # Get the coordinates of the drawing
        x, y = event.x, event.y

        # Draw on the canvas
        r = 12  # Radius of the drawn circle
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

        # Store the drawn strokes for later processing (optional)
        # You can collect multiple strokes to form a complete image for prediction
        self.strokes.append((x, y))

         # Enable predict button (if needed)
        self.predict_btn.config(state=tk.NORMAL)


    def create_image_from_strokes(self):
        # Create a new image with a white background
        img = Image.new("L", (600, 600), "white")
        draw = ImageDraw.Draw(img)

        # Draw the strokes on the image
        for i in range(1, len(self.strokes)):
            x0, y0 = self.strokes[i - 1]
            x1, y1 = self.strokes[i]
            draw.line([(x0, y0), (x1, y1)], fill="black", width=50)  # Adjust width as needed

        # Resize the image to match the model input size (e.g., 28x28)
        img = img.resize((28, 28))
        # Convert the image to an array and normalize
        img_array = img_to_array(img).reshape(1, 28, 28, 1) / 255.0

        return img_array


    def clear_canvas(self):
        self.canvas.delete("all")
        self.img_array = None
        self.strokes = []
        self.predict_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Prediction: ")
        self.accuracy_label.config(text="Accuracy: ")

# Initialize Tkinter
root = tk.Tk()
app = HandwritingRecognitionApp(root)
root.mainloop()
