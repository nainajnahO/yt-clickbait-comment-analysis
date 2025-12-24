import tkinter as tk
from tkinter import messagebox
from typing import List, Dict
import os
import csv


from PIL import Image, ImageTk


class SlideshowApp:
    def __init__(self, root_arg: tk.Tk, slides: List[Dict[str, str]]):

        self.root = root_arg    # INITIALISE TKINTER
        self.slides = slides    # ALL SLIDES
        self.current_index = 0  # CURRENT INDEX
        self.answers = []       # STORE ANSWER

        # INIT TKINTER ARGUMENTS
        self.root.title("Image Slideshow")
        self.root.geometry("600x500")

        # UI ELEMENTS
        self.image_label = tk.Label(root_arg)
        self.image_label.pack(pady=10, expand=True)

            # QUESTION FORMATTING
        self.question_label = tk.Label(root_arg, text="", font=("Helvetica", 14), wraplength=550)
        self.question_label.pack(pady=50)

        # BIND KEYSTROKES
        self.root.bind("<Key>", self.handle_keypress)

        # START SLIDESHOW
        self.load_slide()

    def load_slide(self):

        # IF THERE ARE MORE SLIDES, LOAD NEXT
        if self.current_index < len(self.slides):

            # LOAD SLIDE
            data = self.slides[self.current_index]

            # DISPLAY QUESTION
            self.question_label.config(text=
                                       "Is this clickbait? Press 'y' or 'n'" +
                                       f"\n\nSlide {self.current_index + 1}/{len(self.slides)}")

            try:
                # LOAD IMAGE
                if os.path.exists(os.path.join("media", data['image_path'])):
                    pil_image = Image.open(os.path.join("media", data['image_path']))

                    # RESIZE IMAGE
                    pil_image.thumbnail((400, 300))
                    photo = ImageTk.PhotoImage(pil_image)

                    # DISPLAY IMAGE
                    self.image_label.config(image=photo, text="")
                    self.image_label.image = photo  # Keep a reference to prevent garbage collection
                else:
                    self.image_label.config(image="", text=f"Image not found!")
            except Exception as e:
                self.image_label.config(image="", text=f"Error loading image:\n{e}")

        # IF THERE ARE NO MORE SLIDES, FINISH
        else:
            self.finish()

    def handle_keypress(self, event):
        # GET INPUT
        answer = event.char.lower()

        # IGNORE INVALID KEYS
        if answer not in ("y", "n"):
            return

        # STORE INPUT WITH METADATA
        self.answers.append({
            "image_path": self.slides[self.current_index]['image_path'],
            "answer": answer
        })

        # UPDATE INDEX
        self.current_index += 1
        self.load_slide()

    def finish(self):

        # SAVE RESULTS
        csv_file = "results.csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "answer"])
            for item in self.answers:
                writer.writerow([item["image_path"], item["answer"]])

        messagebox.showinfo("Complete", f"Slideshow finished! Results saved to {csv_file}")
        self.root.destroy()


if __name__ == "__main__":

    # CONFIG
    MEDIA_FOLDER = "media"
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

    # LOAD IMAGES
    slides_data = []
    if os.path.exists(MEDIA_FOLDER):
        # Sort files to ensure a consistent order
        for filename in sorted(os.listdir(MEDIA_FOLDER)):
            if filename.lower().endswith(VALID_EXTENSIONS):
                slides_data.append({"image_path": filename})

    if not slides_data:
        print(f"No images found in '{MEDIA_FOLDER}' folder.")
    else:
        # START TKINTER
        root = tk.Tk()
        app = SlideshowApp(root, slides_data)
        root.mainloop()
