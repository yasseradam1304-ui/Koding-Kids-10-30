import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

cap = cv2.VideoCapture(0)
frame = None

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

root = tk.Tk()
root.title("AI Vision")
video = tk.Label(root)
video.pack()

threading.Thread(target=camera, daemon=True).start()
update()
root.mainloop()