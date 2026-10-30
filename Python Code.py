import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import torch
from torchvision import models

cap = cv2.VideoCapture(0)
frame = None
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

def camera():
    global frame
    while True:
        ok, img = cap.read()
        if ok:
            frame = img

def update():
    if frame is not None:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((480, 360))
        photo = ImageTk.PhotoImage(img)
        video.config(image=photo)
        video.image = photo
    root.after(30, update)

def analyze():
    global frame

    if frame is None:
        return

    img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(x).softmax(1)

    index = prediction.argmax().item()
    result.config(text=weights.meta["categories"][index])

root = tk.Tk()
root.title("AI Vision")
video = tk.Label(root)
video.pack()
result = tk.Label(root, text="What am I looking at?")
result.pack()
button = tk.Button(root, text="Analyze!", command=lambda: threading.Thread(target=analyze, daemon=True).start())
button.pack()

threading.Thread(target=camera, daemon=True).start()
update()
root.mainloop()