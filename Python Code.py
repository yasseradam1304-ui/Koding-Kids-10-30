import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import torch
from PIL import Image, ImageTk
from torchvision import models


class RoundedCard(tk.Canvas):
    def __init__(
        self,
        parent,
        background,
        style,
        radius=22,
        inset=12,
        padding=12,
        fill_height=False,
        **kwargs,
    ):
        super().__init__(
            parent,
            background=parent.winfo_toplevel().cget("background"),
            highlightthickness=0,
            borderwidth=0,
            **kwargs,
        )
        self.card_color = background
        self.radius = radius
        self.inset = inset
        self.fill_height = fill_height
        self.content = ttk.Frame(self, style=style, padding=padding)
        self._content_window = self.create_window(
            inset, inset, anchor="nw", window=self.content
        )
        self.bind("<Configure>", self._resize_card)

    def _resize_card(self, event):
        width = max(event.width, 2 * self.inset + 1)
        height = max(event.height, 2 * self.inset + 1)
        radius = min(self.radius, (width - 2) / 2, (height - 2) / 2)
        x1, y1, x2, y2 = 1, 1, width - 1, height - 1
        points = (
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        )
        if hasattr(self, "_shape"):
            self.coords(self._shape, *points)
        else:
            self._shape = self.create_polygon(
                *points,
                smooth=True,
                splinesteps=24,
                fill=self.card_color,
                outline=self.card_color,
            )
            self.tag_lower(self._shape)
        self.coords(self._content_window, self.inset, self.inset)
        options = {"width": max(1, width - 2 * self.inset)}
        if self.fill_height:
            options["height"] = max(1, height - 2 * self.inset)
        self.itemconfigure(self._content_window, **options)


class AIVisionApp:
    BACKGROUND = "#000000"
    PANEL = "#111111"
    PREVIEW_BACKGROUND = "#050505"
    TEXT = "#ffffff"
    MUTED = "#dedede"
    ACCENT = "#ffff00"
    ACCENT_DARK = "#a6a600"
    HISTORY_PANEL = "#1a1a00"

    def __init__(self, root, capture, model, weights, preprocess):
        self.root = root
        self.capture = capture
        self.model = model
        self.weights = weights
        self.preprocess = preprocess
        self.frame_lock = threading.Lock()
        self.camera_failed = threading.Event()
        self.frame = None
        self.analysis_history = []
        self.history_window = None
        self.history_table = None
        self.camera_available = capture.isOpened()
        self.analysis_running = False
        self.closing = False

        self._configure_window()
        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if self.camera_available:
            threading.Thread(target=self._camera_loop, daemon=True).start()
            self._set_status("Ready", self.ACCENT)
        else:
            self._set_camera_unavailable()

        self._update_preview()

    def _configure_window(self):
        self.root.title("AI Vision – Object Recognition")
        self.root.geometry("960x680")
        self.root.minsize(720, 500)
        self.root.configure(background=self.BACKGROUND)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=self.BACKGROUND)
        style.configure("Card.TFrame", background=self.PANEL)
        style.configure("History.TFrame", background=self.HISTORY_PANEL)
        style.configure(
            "Title.TLabel",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("Segoe UI", 19, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "HistoryTitle.TLabel",
            background=self.HISTORY_PANEL,
            foreground=self.ACCENT,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "HistoryText.TLabel",
            background=self.HISTORY_PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.Card.TLabel",
            background=self.PANEL,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Prediction.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Confidence.TLabel",
            background=self.PANEL,
            foreground="#00ffff",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=self.BACKGROUND,
            foreground=self.ACCENT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background=self.ACCENT,
            foreground="#000000",
            borderwidth=1,
            padding=(12, 7),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#ffffff"), ("disabled", "#404040")],
            foreground=[("disabled", "#bdbdbd")],
        )
        style.configure(
            "Confidence.Horizontal.TProgressbar",
            troughcolor="#263249",
            background=self.ACCENT,
            bordercolor=self.PANEL,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT_DARK,
        )
        style.configure(
            "Exit.TButton",
            background="#252525",
            foreground=self.TEXT,
            borderwidth=1,
            padding=(12, 7),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Exit.TButton", background=[("active", "#454545")])

    def _build_interface(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        app = ttk.Frame(self.root, style="App.TFrame", padding=(16, 12))
        app.grid(row=0, column=0, sticky="nsew")
        app.columnconfigure(0, weight=1)
        app.rowconfigure(1, weight=1)

        header = ttk.Frame(app, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="AI Vision – Object Recognition",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Live camera classification powered by MobileNetV3 Small",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.status_label = ttk.Label(header, text="", style="Status.TLabel")
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))

        preview_card = RoundedCard(
            app,
            self.PANEL,
            "Card.TFrame",
            radius=16,
            inset=8,
            padding=8,
            fill_height=True,
            height=240,
        )
        preview_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        preview_content = preview_card.content
        preview_content.columnconfigure(0, weight=1)
        preview_content.rowconfigure(1, weight=1)

        ttk.Label(
            preview_content, text="Camera preview", style="Section.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.video_label = tk.Label(
            preview_content,
            text="Waiting for camera...",
            background=self.PREVIEW_BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 12),
            highlightthickness=0,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        result_card = ttk.Frame(app, style="Card.TFrame", padding=10)
        result_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        result_card.columnconfigure(0, weight=1)

        ttk.Label(
            result_card, text="Recognition result", style="Section.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.result_label = ttk.Label(
            result_card, text="No analysis yet", style="Prediction.TLabel"
        )
        self.result_label.grid(row=1, column=0, sticky="w", pady=(9, 0))
        self.confidence_label = ttk.Label(
            result_card,
            text="Start an analysis to identify an object.",
            style="Confidence.TLabel",
        )
        self.confidence_label.grid(row=1, column=1, sticky="e", padx=(16, 0))
        self.confidence_bar = ttk.Progressbar(
            result_card,
            style="Confidence.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100,
            length=220,
        )
        self.confidence_bar.grid(
            row=2, column=1, sticky="e", padx=(16, 0), pady=(8, 0)
        )
        self.alternatives_label = ttk.Label(
            result_card,
            text="",
            style="Muted.Card.TLabel",
            wraplength=900,
            justify="left",
        )
        self.alternatives_label.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        controls = ttk.Frame(app, style="App.TFrame")
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        self.analyze_button = ttk.Button(
            controls,
            text="Start Analysis",
            style="Primary.TButton",
            command=self.start_analysis,
        )
        self.analyze_button.grid(row=0, column=1, sticky="e")
        ttk.Button(
            controls,
            text="History",
            style="Exit.TButton",
            command=self.show_history,
        ).grid(row=0, column=2, sticky="e", padx=(10, 0))
        ttk.Button(
            controls,
            text="Exit",
            style="Exit.TButton",
            command=self.close,
        ).grid(row=0, column=3, sticky="e", padx=(10, 0))

    def show_history(self):
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.deiconify()
            self.history_window.lift()
            return

        self.history_window = tk.Toplevel(self.root)
        self.history_window.title("AI Vision - Analysis History")
        self.history_window.geometry("560x360")
        self.history_window.minsize(420, 240)
        self.history_window.configure(background=self.BACKGROUND)
        self.history_window.columnconfigure(0, weight=1)
        self.history_window.rowconfigure(1, weight=1)

        ttk.Label(
            self.history_window,
            text="Analysis History",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 12))

        table_frame = RoundedCard(
            self.history_window,
            self.PANEL,
            "Card.TFrame",
            padding=10,
        )
        table_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        table_content = table_frame.content
        table_content.columnconfigure(0, weight=1)
        table_content.rowconfigure(0, weight=1)

        style = ttk.Style(self.root)
        style.configure(
            "History.Treeview",
            background=self.PANEL,
            fieldbackground=self.PANEL,
            foreground=self.TEXT,
            rowheight=28,
        )
        style.configure(
            "History.Treeview.Heading",
            background=self.HISTORY_PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        self.history_table = ttk.Treeview(
            table_content,
            columns=("time", "object", "confidence"),
            show="headings",
            style="History.Treeview",
        )
        self.history_table.heading("time", text="Time")
        self.history_table.heading("object", text="Recognized object")
        self.history_table.heading("confidence", text="Confidence")
        self.history_table.column("time", width=90, stretch=False)
        self.history_table.column("object", width=280)
        self.history_table.column("confidence", width=110, stretch=False)
        self.history_table.grid(row=0, column=0, sticky="nsew")
        self._refresh_history_table()
        self.history_window.protocol(
            "WM_DELETE_WINDOW", self._close_history_window
        )

    def _refresh_history_table(self):
        if self.history_table is None or not self.history_table.winfo_exists():
            return
        for item in self.history_table.get_children():
            self.history_table.delete(item)
        for recorded_at, label, confidence in reversed(self.analysis_history):
            self.history_table.insert(
                "", "end", values=(recorded_at, label, f"{confidence:.1%}")
            )

    def _close_history_window(self):
        if self.history_window is not None:
            self.history_window.destroy()
        self.history_window = None
        self.history_table = None

    def _set_status(self, text, color):
        self.status_label.configure(text=text, foreground=color)

    def _set_camera_unavailable(self):
        self.camera_available = False
        self._set_status("Camera unavailable", "#ff6666")
        self.video_label.configure(image="", text="Camera unavailable")
        self.video_label.image = None
        if not self.analysis_running:
            self.analyze_button.configure(state=tk.DISABLED)

    def _camera_loop(self):
        while not self.closing:
            ok, image = self.capture.read()
            if not ok:
                if not self.closing:
                    self.camera_failed.set()
                break
            with self.frame_lock:
                self.frame = image

    def _update_preview(self):
        if self.closing:
            return

        if self.camera_failed.is_set() and self.camera_available:
            self._set_camera_unavailable()

        with self.frame_lock:
            image = None if self.frame is None else self.frame.copy()

        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()
        if image is not None and width > 1 and height > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(image)
            preview.thumbnail((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
        elif self.camera_available:
            self.video_label.configure(text="Waiting for camera...")

        self.root.after(30, self._update_preview)

    def start_analysis(self):
        if self.analysis_running:
            return
        if not self.camera_available:
            self._set_camera_unavailable()
            return

        with self.frame_lock:
            image = None if self.frame is None else self.frame.copy()
        if image is None:
            self._set_status("Waiting for camera...", "#f2c879")
            return

        self.analysis_running = True
        self.analyze_button.configure(state=tk.DISABLED)
        self._set_status("Analyzing...", self.ACCENT)
        self.result_label.configure(text="Analyzing image...")
        self.confidence_label.configure(text="")
        self.confidence_bar.configure(value=0)
        self.alternatives_label.configure(text="")
        threading.Thread(
            target=self._analyze_frame, args=(image,), daemon=True
        ).start()

    def _analyze_frame(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            tensor = self.preprocess(image).unsqueeze(0)

            with torch.no_grad():
                probabilities = self.model(tensor).softmax(1)
                scores, indices = probabilities.topk(3, dim=1)

            categories = self.weights.meta["categories"]
            results = [
                (categories[index.item()], score.item())
                for index, score in zip(indices[0], scores[0])
            ]
            self.root.after(0, self._analysis_completed, results)
        except Exception as error:
            self.root.after(0, self._analysis_failed, str(error))

    def _analysis_completed(self, results):
        if self.closing:
            return
        self.analysis_running = False
        self.analyze_button.configure(
            state=tk.NORMAL if self.camera_available else tk.DISABLED
        )
        label, confidence = results[0]
        self.result_label.configure(text=label)
        self.confidence_label.configure(text=f"{confidence:.1%} confidence")
        self.confidence_bar.configure(value=confidence * 100)
        alternatives = "Other predictions: " + "  ·  ".join(
            f"{name} ({score:.1%})" for name, score in results[1:]
        )
        self.alternatives_label.configure(text=alternatives)
        recorded_at = datetime.now().strftime("%H:%M:%S")
        self.analysis_history.append((recorded_at, label, confidence))
        self._refresh_history_table()
        self._set_status("Analysis completed", "#a8ff60")

    def _analysis_failed(self, error):
        if self.closing:
            return
        self.analysis_running = False
        self.analyze_button.configure(
            state=tk.NORMAL if self.camera_available else tk.DISABLED
        )
        self._set_status("Analysis failed", "#ff6666")
        self.result_label.configure(text="Unable to complete analysis")
        self.confidence_label.configure(text="")
        self.confidence_bar.configure(value=0)
        messagebox.showerror("Analysis error", error, parent=self.root)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.capture.release()
        self.root.destroy()


def main():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights).eval()
    preprocess = weights.transforms()
    capture = cv2.VideoCapture(0)

    root = tk.Tk()
    AIVisionApp(root, capture, model, weights, preprocess)
    root.mainloop()


if __name__ == "__main__":
    main()
