
import os
import pandas as pd
import numpy as np
import gensim
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from enriched_sublayer_keywords import enriched_sublayer_keywords
from cta_to_sublayers import cta_to_sublayers

# CTA keyword seeds
cta_seeds = {
    "Advanced Computing and Software": ["computing", "cloud", "algorithms"],
    "Advanced Materials": ["materials", "composites", "nanotech"],
    "Biotechnology": ["biotech", "genomics", "biosensors"],
    "Directed Energy": ["laser", "microwave", "beam"],
    "Future Generation Wireless Technology": ["5g", "wireless", "telecom"],
    "Human-Machine Interfaces": ["interface", "wearables", "exoskeleton"],
    "Hypersonics": ["hypersonic", "scramjet", "aerodynamics"],
    "Integrated Network Systems-of-Systems": ["network", "systems", "command"],
    "Integrated Sensing and Cyber": ["cyber", "sensing", "radar"],
    "Microelectronics": ["semiconductor", "chip", "microelectronics"],
    "Quantum Science": ["quantum", "qubit", "entanglement"],
    "Space Technology": ["satellite", "orbit", "spacecraft"],
    "Trusted AI and Autonomy": ["ai", "autonomy", "robotics"]
}

# GUI logic
class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Classifier GUI")
        self.root.geometry("850x600")

        self.setup_widgets()
        self.model = None
        self.results_df = None

    def setup_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Word2Vec Model Path:").grid(row=0, column=0, padx=5, pady=5)
        self.model_entry = tk.Entry(frame, width=60)
        self.model_entry.grid(row=0, column=1, padx=5)
        tk.Button(frame, text="Browse", command=self.load_model).grid(row=0, column=2, padx=5)

        tk.Label(frame, text="Upload CSV:").grid(row=1, column=0, padx=5)
        tk.Button(frame, text="Browse CSV", command=self.load_csv).grid(row=1, column=1, padx=5, pady=5)

        self.tree = ttk.Treeview(self.root, columns=("Account Name", "CTAs", "Sublayers"), show="headings")
        for col in ("Account Name", "CTAs", "Sublayers"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=250)
        self.tree.pack(expand=True, fill="both", padx=10, pady=10)

        self.save_btn = tk.Button(self.root, text="Save Results", command=self.save_results, state="disabled")
        self.save_btn.pack(pady=5)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Word2Vec Models", "*.bin")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            messagebox.showinfo("Model Loaded", "Word2Vec model loaded successfully!")

    def expand_keywords(self, seeds, topn=5):
        expanded = {}
        for label, words in seeds.items():
            keywords = set(words)
            for word in words:
                if word in self.model:
                    keywords.update([w for w, _ in self.model.most_similar(word, topn=topn)])
            expanded[label] = list(keywords)
        return expanded

    def text_to_vec(self, text):
        words = [w for w in text.lower().split() if w not in ENGLISH_STOP_WORDS and w in self.model]
        if not words:
            return np.zeros(300)
        return np.mean([self.model[w] for w in words], axis=0)

    def load_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            return
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='ISO-8859-1')

        cta_keywords = self.expand_keywords(cta_seeds)
        sublayer_keywords = self.expand_keywords(enriched_sublayer_keywords)

        results = []
        for _, row in df.iterrows():
            desc = str(row.get("Description", ""))
            desc_vec = self.text_to_vec(desc)
            matched_ctas = [cta for cta, words in cta_keywords.items() if any(w in desc.lower() for w in words)]

            matched_sublayers = []
            for cta in matched_ctas:
                sublayers = cta_to_sublayers.get(cta, [])
                for sub in sublayers:
                    if sub not in sublayer_keywords:
                        continue
                    sub_vec = self.text_to_vec(" ".join(sublayer_keywords[sub]))
                    sim = cosine_similarity([desc_vec], [sub_vec])[0][0]
                    if sim > 0.2:
                        matched_sublayers.append(sub)

            results.append({
                "Account Name": row.get("Account Name", ""),
                "Description": desc,
                "CTAs": ", ".join(matched_ctas) if matched_ctas else "UNCLASSIFIED",
                "Sublayers": ", ".join(matched_sublayers) if matched_sublayers else "None"
            })

        self.results_df = pd.DataFrame(results)
        self.tree.delete(*self.tree.get_children())
        for _, row in self.results_df.iterrows():
            self.tree.insert("", "end", values=(row["Account Name"], row["CTAs"], row["Sublayers"]))
        self.save_btn.config(state="normal")
        messagebox.showinfo("Success", "Classification complete!")

    def save_results(self):
        if self.results_df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if path:
                self.results_df.to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Results saved to {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierApp(root)
    root.mainloop()
