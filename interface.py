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
        self.root.geometry("1920x1080")

        # Load and set the background image
        self.background_image = Image.open("background_image.jpg") 
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        # Initialize variables
        self.selected_option = tk.StringVar()
        self.img_array = None
        self.strokes = []
        self.drawing = False

        self.create_widgets()

    def create_widgets(self):
        # Frame for options
        self.option_frame = tk.Frame(self.root, bg='#bae1ff', relief='solid')
        self.option_frame.place(relx=0.45, rely=0.18, anchor = 'n')

        #Radio buttons for options
        self.upload_radio = tk.Radiobutton(self.option_frame, text="Upload Image", variable=self.selected_option, value="upload", command=self.upload_image, font=("Arial", 20), bg='#bae1ff', bd = 5, relief = 'solid')
        self.upload_radio.pack(side=tk.LEFT)

        self.draw_radio = tk.Radiobutton(self.option_frame, text="Draw on Canvas", variable=self.selected_option, value="draw", font=("Arial", 20), bg='#bae1ff', bd = 5, relief = 'solid')
        self.draw_radio.pack(side=tk.LEFT)

        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, bg="white", bd=2, relief="ridge")
        self.canvas.place(relx=0.25, rely=0.23, relwidth=0.4, relheight=0.55) 
        self.canvas_width = self.canvas.winfo_reqwidth()
        self.canvas_height = self.canvas.winfo_reqheight()

        # Draw a bounding box on the canvas to guide the user
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.bounding_box = self.canvas.create_rectangle(10, 10, canvas_width - 10, canvas_height - 10, outline="red")
  
        self.result_frame = tk.Frame(self.root, bg='#ffffff', bd=5, relief='solid')
        self.result_frame.place(relx=0.70, rely=0.4)
        self.result_label = tk.Label(self.result_frame, text="Prediction: ", font=("Comic Sans MS", 20), bg='#bae1ff')
        self.result_label.pack()

        self.accuracy_frame = tk.Frame(self.root, bg='#ffffff', bd=5, relief='solid')
        self.accuracy_frame.place(relx=0.70, rely=0.5)
        self.accuracy_label = tk.Label(self.accuracy_frame, text="Accuracy: ", font=("Comic Sans MS", 20), bg='#bae1ff')
        self.accuracy_label.pack()

        # Clear Canvas button
        self.clear_btn = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas, font=("Comic Sans MS", 18), bg='#bae1ff', bd=5, relief='solid')
        self.clear_btn.place(relx=0.30, rely=0.78, relwidth=0.15, relheight=0.05)

        # Predict button
        self.predict_btn = tk.Button(self.root, text="Predict", command=self.predict, font=("Comic Sans MS", 18), bg='#bae1ff', bd=5, relief='solid')
        self.predict_btn.place(relx=0.45, rely=0.78, relwidth=0.15, relheight=0.05)

        # Bind mouse events for canvas drawing
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

       # Bind resize event
        self.root.bind("<Configure>", self.on_resize)

    def start_drawing(self, event):
        self.drawing = True
        self.strokes.append((event.x, event.y))

    def stop_drawing(self, event):
        self.drawing = False


    def on_resize(self, event):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.coords(self.bounding_box, canvas_width*0.05, canvas_height*0.05, canvas_width*0.95, canvas_height*0.95)
        self.background_photo = ImageTk.PhotoImage(self.background_image.resize((event.width, event.height)))
        self.background_label.config(image=self.background_photo)


    def predict(self):
        if self.selected_option.get() == "upload" and self.img_array is not None:
            prediction = cnn_model.predict(self.img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_class])[0]
            accuracy = prediction[0][predicted_class]
            self.result_label.config(text=f"Prediction: {predicted_label}")
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
        elif self.selected_option.get() == "draw":
            if self.strokes:
                drawn_image = self.create_image_from_strokes()
                
                #UNCOMMENT THIS PART TO SEE WHAT IMAGE IS BEING GENERATED FROM THE IMAGE ARRAY FORMED FROM THOSE STROKES DRAWN BY US IN THE CANVAS PART
                img = Image.fromarray((drawn_image.reshape(28, 28) * 255).astype(np.uint8), mode='L')
                img.show()

                prediction = cnn_model.predict(drawn_image)
                predicted_class = np.argmax(prediction)
                predicted_label = le.inverse_transform([predicted_class])[0]
                accuracy = prediction[0][predicted_class]
                self.result_label.config(text=f"Prediction: {predicted_label}")
                self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")
        else:
            messagebox.showerror("Error", "Please select an option and either upload an image or draw on the canvas.")

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        self.img = load_img(file_path, color_mode='grayscale', target_size=(28, 28))
        self.img_array = img_to_array(self.img).reshape(1, 28, 28, 1) / 255.0
        self.display_img = self.img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.photo_img = ImageTk.PhotoImage(self.display_img)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER, image=self.photo_img)
        self.bounding_box = self.canvas.create_rectangle(self.canvas.winfo_width()*0.05, self.canvas.winfo_height()*0.05, self.canvas.winfo_width()*0.95, self.canvas.winfo_height()*0.95, outline="red")
        self.predict_btn.config(state=tk.NORMAL)

    def draw_on_canvas(self, event):
        if self.drawing:
            x, y = event.x, event.y
            r = 12
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
            self.strokes.append((x, y))
            self.predict_btn.config(state=tk.NORMAL)

    def create_image_from_strokes(self):
        img = Image.new("L", (600, 600), "white")
        draw = ImageDraw.Draw(img)
        for i in range(1, len(self.strokes)):
            x0, y0 = self.strokes[i - 1]
            x1, y1 = self.strokes[i]
            draw.line([(x0, y0), (x1, y1)], fill="black", width=50)
        img = img.crop((60, 60, 540, 540))
        img = img.resize((28, 28))
        img_array = img_to_array(img).reshape(1, 28, 28, 1) / 255.0
        return img_array

    def clear_canvas(self):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()  
        canvas_height = self.canvas.winfo_height()  
        self.bounding_box = self.canvas.create_rectangle(canvas_width*0.05, canvas_height*0.05, canvas_width*0.95, canvas_height*0.95, outline="red")
        self.img_array = None
        self.strokes = []
        self.predict_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Prediction: ")
        self.accuracy_label.config(text="Accuracy: ")

# Initialize Tkinter
root = tk.Tk()
app = HandwritingRecognitionApp(root)
root.mainloop()
