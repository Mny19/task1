import cv2
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import model_from_json

# Function to perform wrinkle detection
def WrinkleDetection(image_path, model):
    face_cascade = cv2.CascadeClassifier("C:\\Users\\NEW\\Downloads\\Wrinkles-Detection-master\\Wrinkles-Detection-master\\haarcascade_frontalface_Default.xml")

    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)

    for x, y, w, h in faces:
        cropped_img = img[y:y + h, x:x + w]
        edges = cv2.Canny(cropped_img, 130, 1000)
        number_of_edges = np.count_nonzero(edges)

    if number_of_edges > 1000:
        return "Wrinkle Found"
    else:
        return "No Wrinkle Found"

def load_model():
    json_file = open("model_a.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_weights.h5")
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model

def DetectWrinkle(file_path, model):
    try:
        result = WrinkleDetection(file_path, model)
        label_wrinkle_result.configure(foreground="#011638", text=result)

    except Exception as e:
        label_wrinkle_result.configure(foreground="#011638", text=f"Error: {str(e)}")

def show_DetectWrinkle_button(file_path, model):
    detect_b = Button(top, text="Detect Wrinkle", command=lambda: DetectWrinkle(file_path, model), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image(model):
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label_wrinkle_result.configure(text='')
        show_DetectWrinkle_button(file_path, model)
    except:
        pass

# Load the pre-trained model
loaded_model = load_model()

top = Tk()
top.geometry('800x600')
top.title('Wrinkle Detector')
top.configure(background='#CDCDCD')

label_wrinkle_result = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

upload = Button(top, text="Upload Image", command=lambda: upload_image(loaded_model), padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label_wrinkle_result.pack(side='bottom', expand='True')
heading = Label(top, text='Wrinkle Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
