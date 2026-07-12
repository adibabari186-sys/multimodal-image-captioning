import os
import sys
import time
import sqlite3
from datetime import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import io

# High-Performance UI, Math & AI Processing Libraries
import customtkinter as ctk
from PIL import Image, ImageStat, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Matplotlib integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Global context variables for Async Boot
processor = None
caption_model = None
det_model = None
models_loaded = False
device = "cpu"

# =========================================================================
# 1. ADVANCED DATABASE LAYER WITH POWER FILTERS
# =========================================================================
class DatabaseManager:
    def __init__(self, db_path="neonvision_enterprise.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    caption TEXT,
                    tags TEXT,
                    sentiment TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()

    def log_caption(self, image_path, caption, tags, sentiment):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO history (image_path, caption, tags, sentiment, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, caption, tags, sentiment, timestamp))
            conn.commit()

    def search_history(self, query):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if not query.strip():
                cursor.execute("SELECT * FROM history ORDER BY id DESC")
            else:
                cursor.execute("SELECT * FROM history WHERE caption LIKE ? OR tags LIKE ? ORDER BY id DESC", (f'%{query}%', f'%{query}%'))
            return cursor.fetchall()


# =========================================================================
# 2. BACKGROUND MODEL INITIALIZATION BLOCK
# =========================================================================
def load_ai_models_async(status_callback):
    global processor, caption_model, det_model, models_loaded, device
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        status_callback(f"🧠 Aligning Pipelines [{device.upper()}]...")
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        status_callback("🎯 Tuning YOLO Anchor Grids...")
        det_model = YOLO("yolov8n.pt")
        
        models_loaded = True
        status_callback("🟢 ENTERPRISE CORE: ONLINE")
    except Exception as e:
        status_callback("🔴 WORKFLOW INITIALIZATION FAULT")
        print(f"Error compiling neural weights: {e}")


# =========================================================================
# 3. TEXT INTELLIGENCE & HEURISTIC ENGINE (NLP Logic Suite)
# =========================================================================
class TextIntelligence:
    @staticmethod
    def detect_sentiment(text):
        positive_keywords = ["beautiful", "sunny", "happy", "smiling", "bright", "clean", "joy", "scenic", "modern", "vibrant", "man", "woman"]
        negative_keywords = ["dark", "gloomy", "alone", "broken", "dirty", "stormy", "sad", "empty", "old", "ruins"]
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_keywords if word in text_lower)
        neg_score = sum(1 for word in negative_keywords if word in text_lower)
        if pos_score > neg_score: return "🔴 Positive / High Lumens"
        elif neg_score > pos_score: return "🔵 Somber / Dark Mode"
        return "⚪ Neutral Profile"

    @staticmethod
    def expand_caption(text):
        return f"A high-fidelity matrix capture illustrating {text.lower()} optimized via multi-model token inference layer."

    @staticmethod
    def local_translate(text, target_lang):
        translations = {
            "hindi": "यह दृश्य दर्शाता है: " + text.replace("A ", "").replace("a ", ""),
            "urdu": "یہ منظر دکھاتی ہے: " + text.replace("A ", "").replace("a ", "")
        }
        return translations.get(target_lang, text)

    @staticmethod
    def deep_summarize(text):
        sentences = text.split('.')
        clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(clean_sentences) <= 2:
            return f"💡 Context Node: {text.strip()}"
        return f"🔹 Abstract: {clean_sentences[0]}.\n\n🔹 Takeaway: {clean_sentences[-1]}."

    @staticmethod
    def generate_image_from_text(prompt):
        import requests
        cleaned_prompt = prompt.replace(" ", ",")
        url = f"https://image.pollinations.ai/p/{cleaned_prompt}?width=500&height=500&nologo=true"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Art gen timeout: {e}")
        return None


# =========================================================================
# 4. CUSTOM ELEMENTS SUITE
# =========================================================================
class NeonButton(ctk.CTkButton):
    def __init__(self, master, text, **kwargs):
        super().__init__(
            master, text=text, corner_radius=6, border_width=2,
            border_color="#00ffff", fg_color="#0f001e", hover_color="#ff007f",
            text_color="#ffffff", font=ctk.CTkFont(family="Courier", size=12, weight="bold"),
            **kwargs
        )

class GlassFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=12, fg_color="#070010", border_width=1, border_color="#ff007f", **kwargs)


# =========================================================================
# 5. CORE WORKSPACE (DASHBOARD SUITE)
# =========================================================================
class DashboardView(ctk.CTkFrame):
    def __init__(self, master, db_manager):
        super().__init__(master, fg_color="transparent")
        self.db = db_manager
        self.selected_file = None
        self.original_img = None
        self.last_analysis_results = None
        self.chart_canvas = None

        self.grid_columnconfigure((0, 1), weight=1, uniform="equal")
        self.grid_rowconfigure(0, weight=1)

        # --------- LEFT PANEL: INPUT VIEW & FX ENGINE ---------
        self.left_panel = GlassFrame(self)
        self.left_panel.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        
        self.preview_lbl = ctk.CTkLabel(self.left_panel, text="[ SYSTEM ONLINE // READY TO INJECT SOURCE ]", font=ctk.CTkFont(family="Courier", size=12), text_color="#00ffff")
        self.preview_lbl.pack(expand=True, fill="both", padx=20, pady=10)

        # DSP Filter Layer
        self.filter_dock = ctk.CTkFrame(self.left_panel, fg_color="#030008", height=55)
        self.filter_dock.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(self.filter_dock, text="DSP MATRIX:", font=ctk.CTkFont(family="Courier", size=10), text_color="#ff007f").pack(side="left", padx=10)
        ctk.CTkButton(self.filter_dock, text="RAW", width=55, command=lambda: self.apply_dsp_filter("raw")).pack(side="left", padx=3)
        ctk.CTkButton(self.filter_dock, text="GRAY", width=55, command=lambda: self.apply_dsp_filter("gray")).pack(side="left", padx=3)
        ctk.CTkButton(self.filter_dock, text="BLUR", width=55, command=lambda: self.apply_dsp_filter("blur")).pack(side="left", padx=3)

        self.btn_upload = NeonButton(self.left_panel, text="LOAD IMAGE SOURCE OVERLAY", command=self.upload_image)
        self.btn_upload.pack(pady=15, side="bottom", fill="x", padx=35)

        # --------- RIGHT PANEL: MULTI-TAB MATRIX SUITE ---------
        self.right_panel = GlassFrame(self)
        self.right_panel.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")

        self.tab_container = ctk.CTkTabview(self.right_panel, fg_color="#030008", segmented_button_selected_color="#ff007f", segmented_button_unselected_color="#140026")
        self.tab_container.pack(fill="both", expand=True, padx=15, pady=10)
        self.tab_container.add("Primary Vector")
        self.tab_container.add("Telemetry Chart")
        self.tab_container.add("Text-to-AI Matrix")
        self.tab_container.add("Cross Translation")

        # Tab 1: Vision Output Frame
        self.caption_out = ctk.CTkTextbox(self.tab_container.tab("Primary Vector"), font=ctk.CTkFont(size=14), wrap="word", border_width=1, border_color="#00ffff")
        self.caption_out.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.control_ribbon = ctk.CTkFrame(self.tab_container.tab("Primary Vector"), fg_color="transparent")
        self.control_ribbon.pack(fill="x", pady=4)
        ctk.CTkButton(self.control_ribbon, text="📋 Copy", width=70, command=self.copy_to_clipboard).pack(side="left", padx=2)
        ctk.CTkButton(self.control_ribbon, text="✨ Expand", width=85, command=self.expand_current_caption).pack(side="left", padx=2)
        ctk.CTkButton(self.control_ribbon, text="🔊 Voice", width=75, command=self.speak_caption).pack(side="left", padx=2)
        ctk.CTkButton(self.control_ribbon, text="📄 Compile PDF", width=100, command=self.export_pdf_report).pack(side="right", padx=2)

        # Tab 2: Matplotlib Real-time Analytics Visualizer
        self.chart_container = ctk.CTkFrame(self.tab_container.tab("Telemetry Chart"), fg_color="transparent")
        self.chart_container.pack(fill="both", expand=True)
        self.empty_chart_lbl = ctk.CTkLabel(self.chart_container, text="[ RE-ROUTE PIPELINE TO COMPUTE ANALYTICS PLOT ]", font=ctk.CTkFont(family="Courier", size=11))
        self.empty_chart_lbl.pack(expand=True)

        # Tab 3: Text to Reverse Engine & Summary Modules
        self.text_ai_frame = ctk.CTkFrame(self.tab_container.tab("Text-to-AI Matrix"), fg_color="transparent")
        self.text_ai_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(self.text_ai_frame, text="INPUT SOURCE TEXT / ARCHIVE LAYER PROMPT:", font=ctk.CTkFont(family="Courier", size=11), text_color="#00ffff").pack(anchor="w")
        self.user_text_input = ctk.CTkTextbox(self.text_ai_frame, height=90, border_width=1, border_color="#ff007f")
        self.user_text_input.pack(fill="x", pady=5)
        
        self.btn_text_ai = NeonButton(self.text_ai_frame, text="🔮 SYNTHESIZE REVERSE TRANSLATION (SUMMARY + ART)", command=self.trigger_text_to_ai_flow)
        self.btn_text_ai.pack(fill="x", pady=5)
        
        self.summary_output_box = ctk.CTkTextbox(self.text_ai_frame, font=ctk.CTkFont(size=12), wrap="word", border_width=1, border_color="#00ffff")
        self.summary_output_box.pack(fill="both", expand=True, pady=5)

        # Tab 4: Localized Cross Language Terminal
        self.trans_box = ctk.CTkTextbox(self.tab_container.tab("Cross Translation"), font=ctk.CTkFont(size=13), wrap="word")
        self.trans_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Core Computation Master Trigger Switch
        self.btn_process = NeonButton(self.right_panel, text="⚡ EXECUTE ARCHITECTURE MULTI-MODEL COMPUTE", command=self.start_process_thread)
        self.btn_process.pack(pady=15, fill="x", padx=30, side="bottom")

    def upload_image(self):
        file = filedialog.askopenfilename(filetypes=[("Image Formats", "*.jpg *.png *.jpeg *.webp")])
        if file:
            self.selected_file = file
            self.original_img = Image.open(file)
            self.refresh_image_display(self.original_img)

    def refresh_image_display(self, pil_img):
        canvas_img = pil_img.copy()
        canvas_img.thumbnail((420, 420))
        ctk_img = ctk.CTkImage(light_image=canvas_img, dark_image=canvas_img, size=canvas_img.size)
        self.preview_lbl.configure(image=ctk_img, text="")
        self.preview_lbl.image = ctk_img

    def apply_dsp_filter(self, matrix_type):
        if not self.original_img: return
        working_matrix = self.original_img.copy()
        if matrix_type == "gray": working_matrix = working_matrix.convert("L")
        elif matrix_type == "blur": working_matrix = working_matrix.filter(ImageFilter.GaussianBlur(6))
        cache_path = "transient_filter_node.jpg"
        working_matrix.convert("RGB").save(cache_path)
        self.selected_file = cache_path
        self.refresh_image_display(working_matrix)

    def start_process_thread(self):
        if not models_loaded: return
        if not self.selected_file: return
        self.btn_process.configure(text="COMPUTING PIPELINE TOKENS...", state="disabled")
        threading.Thread(target=self.process_image, daemon=True).start()

    def process_image(self):
        global processor, caption_model, det_model, device
        try:
            start_timestamp = time.time()
            raw_image = Image.open(self.selected_file).convert('RGB')
            stat = ImageStat.Stat(raw_image)
            brightness_index = sum(stat.mean) / 3
            
            inputs = processor(raw_image, return_tensors="pt").to(device)
            generated_tokens = caption_model.generate(**inputs)
            base_caption = processor.decode(generated_tokens[0], skip_special_tokens=True).capitalize()

            localization_tracks = det_model(self.selected_file)
            detected_entities = [det_model.names[int(box.cls[0])] for box in localization_tracks[0].boxes]
            entity_frequency = {entity: detected_entities.count(entity) for entity in set(detected_entities)}
            if not entity_frequency: entity_frequency = {"Context Arrays": 1}

            sentiment_node = TextIntelligence.detect_sentiment(base_caption)
            tag_anchors = [f"#{e.replace(' ', '')}" for e in entity_frequency.keys()]

            self.caption_out.delete("1.0", "end")
            self.caption_out.insert("1.0", f"🎯 DESCRIPTION:\n{base_caption}\n\n📊 SPECTRUM MATRIX ANALYSIS:\n• Mood Node: {sentiment_node}\n• Luminous Density Vector: {int(brightness_index)}")

            self.trans_box.delete("1.0", "end")
            self.trans_box.insert("end", f"▶️ HINDI MATRIX DECODE:\n{TextIntelligence.local_translate(base_caption, 'hindi')}\n\n▶️ URDU MATRIX DECODE:\n{TextIntelligence.local_translate(base_caption, 'urdu')}")

            self.last_analysis_results = {"caption": base_caption, "sentiment": sentiment_node, "brightness": int(brightness_index)}
            self.master.after(0, self.plot_telemetry_metrics, entity_frequency)
            self.db.log_caption(self.selected_file, base_caption, " ".join(tag_anchors[:6]), sentiment_node)
        except Exception as e:
            print(e)
        finally:
            self.btn_process.configure(text="⚡ EXECUTE ARCHITECTURE MULTI-MODEL COMPUTE", state="normal")

    def trigger_text_to_ai_flow(self):
        user_raw_payload = self.user_text_input.get("1.0", "end-1c").strip()
        if not user_raw_payload: return
        self.btn_text_ai.configure(text="COMPILING CONTEXT MATRIX UNIVERSE...", state="disabled")
        threading.Thread(target=self.execute_text_ai_processing, args=(user_raw_payload,), daemon=True).start()

    def execute_text_ai_processing(self, raw_text):
        try:
            derived_summary = TextIntelligence.deep_summarize(raw_text)
            rendered_art_pil = TextIntelligence.generate_image_from_text(raw_text)
            self.master.after(0, self.finalize_text_ai_graphics, derived_summary, rendered_art_pil)
        except Exception as e:
            print(e)
        finally:
            self.master.after(0, lambda: self.btn_text_ai.configure(text="🔮 SYNTHESIZE REVERSE TRANSLATION (SUMMARY + ART)", state="normal"))

    def finalize_text_ai_graphics(self, summary_text, pil_image):
        self.summary_output_box.delete("1.0", "end")
        self.summary_output_box.insert("1.0", f"📝 COGNITIVE METADATA SUMMARY:\n\n{summary_text}")
        if pil_image:
            self.original_img = pil_image
            self.refresh_image_display(pil_image)

    def plot_telemetry_metrics(self, entity_frequency):
        if self.chart_canvas: self.chart_canvas.get_tk_widget().destroy()
        self.empty_chart_lbl.pack_forget()
        figure_canvas = Figure(figsize=(4.5, 3.2), dpi=100, facecolor='#030008')
        axis = figure_canvas.add_subplot(111, facecolor='#070010')
        axis.bar(list(entity_frequency.keys()), list(entity_frequency.values()), color="#00ffff", edgecolor="#ff007f")
        axis.tick_params(colors='#ffffff', labelsize=8)
        self.chart_canvas = FigureCanvasTkAgg(figure_canvas, master=self.chart_container)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.caption_out.get("1.0", "end-1c"))

    def expand_current_caption(self):
        if self.last_analysis_results:
            self.caption_out.insert("end", f"\n\n✨ TRANSFORMER EXTENSION:\n{TextIntelligence.expand_caption(self.last_analysis_results['caption'])}")

    def speak_caption(self):
        if self.last_analysis_results:
            from gtts import gTTS
            gTTS(text=self.last_analysis_results["caption"], lang='en').save("neon_speech.mp3")
            os.system("start neon_speech.mp3" if sys.platform == "win32" else "open neon_speech.mp3")

    def export_pdf_report(self):
        if not self.last_analysis_results: return
        file_destination = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Document", "*.pdf")])
        if file_destination:
            doc = canvas.Canvas(file_destination, pagesize=letter)
            doc.drawString(45, 750, "=== NEONVISION PRO COMPUTE SUITE LOG ===")
            doc.drawString(45, 720, f"Caption Array: {self.last_analysis_results['caption']}")
            doc.drawString(45, 700, f"Mood Mapping: {self.last_analysis_results['sentiment']}")
            doc.drawString(45, 680, f"Luminous Scalar Metric: {self.last_analysis_results['brightness']}")
            doc.save()
            messagebox.showinfo("Export Success", "Enterprise Ledger serialized to local drive layer.")


# =========================================================================
# 6. HISTORICAL LOG DATA ENGINE
# =========================================================================
class HistoryView(ctk.CTkFrame):
    def __init__(self, master, db_manager):
        super().__init__(master, fg_color="transparent")
        self.db = db_manager
        self.control_dock = ctk.CTkFrame(self, fg_color="#070010", border_width=1, border_color="#00ffff")
        self.control_dock.pack(fill="x", padx=20, pady=12)
        self.search_field = ctk.CTkEntry(self.control_dock, placeholder_text="Enter pipeline tokens to perform historical matrix query lookup...", font=ctk.CTkFont(family="Courier"), fg_color="#030008", border_color="#ff007f")
        self.search_field.pack(side="left", fill="x", expand=True, padx=15, pady=12)
        ctk.CTkButton(self.control_dock, text="🔍 INJECT QUERY", width=130, command=self.refresh_records).pack(side="right", padx=15)
        self.table_box = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Courier", size=12), border_width=1, border_color="#ff007f")
        self.table_box.pack(fill="both", expand=True, padx=20, pady=10)

    def grid(self, **kwargs):
        super().grid(**kwargs)
        self.refresh_records()

    def refresh_records(self):
        self.table_box.delete("1.0", "end")
        data_rows = self.db.search_history(self.search_field.get().strip())
        for data_node in data_rows:
            self.table_box.insert("end", f" 🗓️ LOG TIMESTAMP: [{data_node[5]}]\n 📝 VECTOR TOKENS: {data_node[2]}\n 🎭 MOOD COEFFICIENT: {data_node[4]}\n {'='*85}\n")


# =========================================================================
# 7. SYSTEM APPLICATION ROOT LOGIC
# =========================================================================
class NeonVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NEONVISION ENTERPRISE AI PLATFORM // 11X HEURISTIC INFERENCE GRID ENVIRONMENT")
        self.geometry("1280x810")
        ctk.set_appearance_mode("dark")

        self.db = DatabaseManager()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.initialize_sidebar()
        self.views = {"dashboard": DashboardView(self, self.db), "history": HistoryView(self, self.db)}
        self.show_view("dashboard")
        threading.Thread(target=load_ai_models_async, args=(self.update_hardware_footer,), daemon=True).start()

    def initialize_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=235, corner_radius=0, fg_color="#030006", border_width=1, border_color="#00ffff")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        ctk.CTkLabel(self.sidebar, text="NEONVISION // ENTERPRISE", font=ctk.CTkFont(family="Courier", size=14, weight="bold"), text_color="#00ffff").pack(pady=40)
        ctk.CTkButton(self.sidebar, text="🤖 COGNITIVE WORKSPACE", anchor="w", command=lambda: self.show_view("dashboard")).pack(fill="x", padx=15, pady=8)
        ctk.CTkButton(self.sidebar, text="📜 QUANTUM LEDGER LOGS", anchor="w", command=lambda: self.show_view("history")).pack(fill="x", padx=15, pady=8)
        self.hardware_footer = ctk.CTkLabel(self.sidebar, text="INITIALIZING VECTOR ARRAYS...", font=ctk.CTkFont(family="Courier", size=10), text_color="#ff007f")
        self.hardware_footer.pack(side="bottom", pady=25)

    def update_hardware_footer(self, text):
        self.hardware_footer.configure(text=text)

    def show_view(self, view_name):
        for view in self.views.values(): view.grid_forget()
        self.views[view_name].grid(row=0, column=1, sticky="nsew", padx=25, pady=25)

if __name__ == "__main__":
    app = NeonVisionApp()
    app.mainloop()
