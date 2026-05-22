"""
gui.py - Modern CustomTkinter GUI for Placement Prediction System
=================================================================
Professional dark-mode AI dashboard with sidebar navigation,
student form, prediction results, analytics charts, and records view.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import time
import os
import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from model import predict_placement, get_feature_importance
from database import (
    initialize_database, insert_student, get_all_students,
    search_students, delete_student, get_statistics, get_cgpa_distribution
)

# ── Color Palette ─────────────────────────────────────────────
BG_DARK      = "#0f0f1a"
BG_CARD      = "#1a1a2e"
BG_SIDEBAR   = "#12122b"
ACCENT       = "#6c63ff"
ACCENT_HOVER = "#5a52d5"
SUCCESS      = "#00c896"
DANGER       = "#ff4757"
WARNING      = "#ffa502"
TEXT_PRIMARY  = "#e8e8f0"
TEXT_SECONDARY= "#8888a8"
BORDER       = "#2a2a4a"
INPUT_BG     = "#16162e"

# ── CustomTkinter Settings ────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PlacementApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("AI Placement Prediction System")
        self.geometry("1280x780")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)
        self.current_page = None
        self.logged_in = False

        # Center the window on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 1280) // 2
        y = (self.winfo_screenheight() - 780) // 2 - 30
        self.geometry(f"1280x780+{x}+{y}")

        initialize_database()
        self.show_login()

    # ══════════════════════════════════════════════════════════
    #  LOGIN PAGE
    # ══════════════════════════════════════════════════════════
    def show_login(self):
        self._clear_window()
        frame = ctk.CTkFrame(self, fg_color=BG_DARK)
        frame.pack(fill="both", expand=True)

        # Center card
        card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=20, width=420, height=520, border_width=1, border_color=BORDER)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(expand=True, padx=40, pady=30)

        # Logo icon
        ctk.CTkLabel(inner, text="🎓", font=ctk.CTkFont(size=52)).pack(pady=(0, 5))
        ctk.CTkLabel(inner, text="AI Placement Predictor", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRIMARY).pack()
        ctk.CTkLabel(inner, text="Sign in to continue", font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY).pack(pady=(2, 25))

        # Username
        ctk.CTkLabel(inner, text="Username", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY, anchor="w").pack(fill="x")
        self.login_user = ctk.CTkEntry(inner, height=42, fg_color=INPUT_BG, border_color=BORDER, text_color=TEXT_PRIMARY, placeholder_text="Enter username")
        self.login_user.pack(fill="x", pady=(2, 12))

        # Password
        ctk.CTkLabel(inner, text="Password", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY, anchor="w").pack(fill="x")
        self.login_pass = ctk.CTkEntry(inner, height=42, fg_color=INPUT_BG, border_color=BORDER, text_color=TEXT_PRIMARY, placeholder_text="Enter password", show="●")
        self.login_pass.pack(fill="x", pady=(2, 20))

        # Login button
        btn = ctk.CTkButton(inner, text="Sign In  →", height=44, corner_radius=10, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                            font=ctk.CTkFont(size=14, weight="bold"), command=self._do_login)
        btn.pack(fill="x", pady=(0, 10))

        # Hint
        ctk.CTkLabel(inner, text="Demo: admin / admin", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack()
        self.login_user.bind("<Return>", lambda e: self._do_login())
        self.login_pass.bind("<Return>", lambda e: self._do_login())

    def _do_login(self):
        user = self.login_user.get().strip()
        pwd = self.login_pass.get().strip()
        if user == "admin" and pwd == "admin":
            self.logged_in = True
            self._build_main_ui()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials.\nUse admin / admin")

    # ══════════════════════════════════════════════════════════
    #  MAIN UI SCAFFOLD (sidebar + content)
    # ══════════════════════════════════════════════════════════
    def _build_main_ui(self):
        self._clear_window()
        wrapper = ctk.CTkFrame(self, fg_color=BG_DARK)
        wrapper.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = ctk.CTkFrame(wrapper, fg_color=BG_SIDEBAR, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Sidebar header
        hdr = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(20, 30))
        ctk.CTkLabel(hdr, text="🎓", font=ctk.CTkFont(size=28)).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(hdr, text="PlaceAI", font=ctk.CTkFont(size=18, weight="bold"), text_color=ACCENT).pack(side="left")

        # Nav buttons
        nav_items = [
            ("📊  Dashboard",  self.show_dashboard),
            ("📝  Predict",     self.show_predict),
            ("📋  Records",     self.show_records),
            ("📈  Analytics",   self.show_analytics),
        ]
        self.nav_btns = []
        for label, cmd in nav_items:
            b = ctk.CTkButton(self.sidebar, text=label, anchor="w", height=42, corner_radius=8,
                              fg_color="transparent", hover_color=ACCENT_HOVER, text_color=TEXT_PRIMARY,
                              font=ctk.CTkFont(size=13), command=cmd)
            b.pack(fill="x", padx=12, pady=2)
            self.nav_btns.append(b)

        # Logout at bottom
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        ctk.CTkButton(self.sidebar, text="🚪  Logout", anchor="w", height=40, corner_radius=8,
                       fg_color="transparent", hover_color="#3a1a1a", text_color=DANGER,
                       font=ctk.CTkFont(size=13), command=self.show_login).pack(fill="x", padx=12, pady=(0, 20))

        # Content area
        self.content = ctk.CTkFrame(wrapper, fg_color=BG_DARK, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)

        self.show_dashboard()

    def _set_active_nav(self, idx):
        for i, b in enumerate(self.nav_btns):
            b.configure(fg_color=ACCENT if i == idx else "transparent")

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _clear_window(self):
        for w in self.winfo_children():
            w.destroy()

    # ══════════════════════════════════════════════════════════
    #  DASHBOARD PAGE
    # ══════════════════════════════════════════════════════════
    def show_dashboard(self):
        self._clear_content()
        self._set_active_nav(0)

        scroll = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=18)

        # Title
        ctk.CTkLabel(scroll, text="Dashboard", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(scroll, text="Overview of placement analytics", font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 16))

        stats = get_statistics()

        # Stat cards row
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=(0, 18))
        cards_data = [
            ("Total Students", str(stats["total"]), "👥", ACCENT),
            ("Placed", str(stats["placed"]), "✅", SUCCESS),
            ("Not Placed", str(stats["not_placed"]), "❌", DANGER),
            ("Placement Rate", f"{stats['placement_rate']:.1f}%", "📊", WARNING),
        ]
        for i, (title, val, icon, color) in enumerate(cards_data):
            row.grid_columnconfigure(i, weight=1)
            c = ctk.CTkFrame(row, fg_color=BG_CARD, corner_radius=14, height=110, border_width=1, border_color=BORDER)
            c.grid(row=0, column=i, padx=6, sticky="nsew")
            c.pack_propagate(False)
            inner = ctk.CTkFrame(c, fg_color="transparent")
            inner.pack(expand=True, padx=18)
            ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(size=28)).pack(anchor="w")
            ctk.CTkLabel(inner, text=val, font=ctk.CTkFont(size=26, weight="bold"), text_color=color).pack(anchor="w")
            ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(anchor="w")

        # Second row: avg stats
        row2 = ctk.CTkFrame(scroll, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 18))
        cards2 = [
            ("Avg CGPA", str(stats["avg_cgpa"]), "📚", "#a29bfe"),
            ("Avg Aptitude", str(stats["avg_aptitude"]), "🧠", "#fd79a8"),
            ("Avg Probability", f"{stats['avg_probability']}%", "🎯", "#00cec9"),
        ]
        for i, (title, val, icon, color) in enumerate(cards2):
            row2.grid_columnconfigure(i, weight=1)
            c = ctk.CTkFrame(row2, fg_color=BG_CARD, corner_radius=14, height=100, border_width=1, border_color=BORDER)
            c.grid(row=0, column=i, padx=6, sticky="nsew")
            c.pack_propagate(False)
            inner = ctk.CTkFrame(c, fg_color="transparent")
            inner.pack(expand=True, padx=18)
            ctk.CTkLabel(inner, text=f"{icon}  {title}", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY).pack(anchor="w")
            ctk.CTkLabel(inner, text=val, font=ctk.CTkFont(size=22, weight="bold"), text_color=color).pack(anchor="w")

        # Feature importance chart
        fi = get_feature_importance()
        if fi:
            chart_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
            chart_card.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(chart_card, text="  🔍  Feature Importance", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=18, pady=(14, 4))

            fig = Figure(figsize=(8, 2.8), facecolor=BG_CARD)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            names = list(fi.keys())
            vals = list(fi.values())
            colors_bar = ["#6c63ff","#a29bfe","#fd79a8","#00cec9","#ffa502","#ff6348","#2ed573","#1e90ff"]
            ax.barh(names, vals, color=colors_bar[:len(names)], height=0.55, edgecolor="none")
            ax.set_xlabel("Importance (%)", color=TEXT_SECONDARY, fontsize=9)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout(pad=1.5)
            canvas = FigureCanvasTkAgg(fig, chart_card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x", padx=14, pady=(0, 14))

    # ══════════════════════════════════════════════════════════
    #  PREDICT PAGE
    # ══════════════════════════════════════════════════════════
    def show_predict(self):
        self._clear_content()
        self._set_active_nav(1)

        scroll = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=18)

        ctk.CTkLabel(scroll, text="Placement Prediction", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(scroll, text="Enter student details to predict placement", font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 16))

        # Form card
        form_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        form_card.pack(fill="x", pady=(0, 14))

        grid = ctk.CTkFrame(form_card, fg_color="transparent")
        grid.pack(fill="x", padx=24, pady=20)

        self.entries = {}
        fields = [
            ("Name", "text", "e.g. Rohit Sharma"),
            ("CGPA", "float", "0.0 - 10.0"),
            ("Aptitude Score", "float", "0 - 100"),
            ("Communication Skills", "int", "1 - 10"),
            ("Technical Skills", "int", "1 - 10"),
            ("Internship Count", "int", "0 - 10"),
            ("Project Count", "int", "0 - 20"),
            ("Attendance %", "float", "0 - 100"),
            ("Certifications", "int", "0 - 20"),
        ]
        for idx, (label, dtype, ph) in enumerate(fields):
            r, c = divmod(idx, 3)
            grid.grid_columnconfigure(c, weight=1)
            f = ctk.CTkFrame(grid, fg_color="transparent")
            f.grid(row=r, column=c, padx=8, pady=6, sticky="nsew")
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(anchor="w")
            e = ctk.CTkEntry(f, height=38, fg_color=INPUT_BG, border_color=BORDER, text_color=TEXT_PRIMARY, placeholder_text=ph)
            e.pack(fill="x", pady=(2, 0))
            self.entries[label] = e

        # Buttons
        btn_row = ctk.CTkFrame(form_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(0, 20))
        ctk.CTkButton(btn_row, text="🔮  Predict Placement", height=44, corner_radius=10, fg_color=ACCENT,
                       hover_color=ACCENT_HOVER, font=ctk.CTkFont(size=14, weight="bold"),
                       command=self._run_prediction).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_row, text="🗑  Clear", height=44, corner_radius=10, fg_color="#2a2a4a",
                       hover_color="#3a3a5a", font=ctk.CTkFont(size=13),
                       command=self._clear_form).pack(side="left")

        # Result area
        self.result_frame = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        self.result_frame.pack(fill="x", pady=(0, 10))
        self.result_inner = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        self.result_inner.pack(fill="x", padx=24, pady=20)
        ctk.CTkLabel(self.result_inner, text="🔮  Enter student details and click Predict",
                      font=ctk.CTkFont(size=14), text_color=TEXT_SECONDARY).pack()

    def _clear_form(self):
        for e in self.entries.values():
            e.delete(0, "end")

    def _run_prediction(self):
        try:
            name = self.entries["Name"].get().strip()
            if not name:
                messagebox.showwarning("Input Error", "Please enter student name.")
                return
            cgpa = float(self.entries["CGPA"].get())
            apt = float(self.entries["Aptitude Score"].get())
            comm = int(self.entries["Communication Skills"].get())
            tech = int(self.entries["Technical Skills"].get())
            intr = int(self.entries["Internship Count"].get())
            proj = int(self.entries["Project Count"].get())
            att = float(self.entries["Attendance %"].get())
            cert = int(self.entries["Certifications"].get())
        except ValueError:
            messagebox.showwarning("Input Error", "Please fill all fields with valid numbers.")
            return

        # Show loading
        for w in self.result_inner.winfo_children():
            w.destroy()
        self.progress = ctk.CTkProgressBar(self.result_inner, mode="indeterminate", progress_color=ACCENT)
        self.progress.pack(fill="x", pady=20)
        self.progress.start()
        loading_lbl = ctk.CTkLabel(self.result_inner, text="⏳ Analyzing student profile...", font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY)
        loading_lbl.pack()

        def do_predict():
            time.sleep(1.2)  # Simulate loading
            result = predict_placement(cgpa, apt, comm, tech, intr, proj, att, cert)
            insert_student(name, cgpa, apt, comm, tech, intr, proj, att, cert, result["prediction"], result["probability"])
            self.after(0, lambda: self._show_result(result, name))

        threading.Thread(target=do_predict, daemon=True).start()

    def _show_result(self, result, name):
        for w in self.result_inner.winfo_children():
            w.destroy()

        pred = result["prediction"]
        prob = result["probability"]
        color = SUCCESS if pred == "Placed" else DANGER
        icon = "✅" if pred == "Placed" else "❌"

        # Header
        ctk.CTkLabel(self.result_inner, text=f"{icon}  Prediction Result for {name}",
                      font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        # Prediction + probability row
        top = ctk.CTkFrame(self.result_inner, fg_color="transparent")
        top.pack(fill="x", pady=(0, 10))

        pred_card = ctk.CTkFrame(top, fg_color=BG_DARK, corner_radius=12, border_width=1, border_color=color)
        pred_card.pack(side="left", padx=(0, 12), fill="x", expand=True)
        ctk.CTkLabel(pred_card, text="Status", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(padx=16, pady=(10, 0), anchor="w")
        ctk.CTkLabel(pred_card, text=pred, font=ctk.CTkFont(size=28, weight="bold"), text_color=color).pack(padx=16, pady=(0, 10), anchor="w")

        prob_card = ctk.CTkFrame(top, fg_color=BG_DARK, corner_radius=12, border_width=1, border_color=ACCENT)
        prob_card.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(prob_card, text="Placement Probability", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(padx=16, pady=(10, 0), anchor="w")
        ctk.CTkLabel(prob_card, text=f"{prob}%", font=ctk.CTkFont(size=28, weight="bold"), text_color=ACCENT).pack(padx=16, pady=(0, 4), anchor="w")
        pbar = ctk.CTkProgressBar(prob_card, progress_color=color, height=8)
        pbar.set(prob / 100)
        pbar.pack(fill="x", padx=16, pady=(0, 10))

        # Suggestions
        ctk.CTkLabel(self.result_inner, text="💡 Improvement Suggestions", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", pady=(10, 6))
        for s in result["suggestions"]:
            ctk.CTkLabel(self.result_inner, text=s, font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY, wraplength=700, anchor="w", justify="left").pack(anchor="w", pady=2)

    # ══════════════════════════════════════════════════════════
    #  RECORDS PAGE
    # ══════════════════════════════════════════════════════════
    def show_records(self):
        self._clear_content()
        self._set_active_nav(2)

        top = ctk.CTkFrame(self.content, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(18, 0))
        ctk.CTkLabel(top, text="Student Records", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")

        # Search + Export
        right = ctk.CTkFrame(top, fg_color="transparent")
        right.pack(side="right")
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *a: self._refresh_records())
        ctk.CTkEntry(right, textvariable=self.search_var, width=220, height=36, fg_color=INPUT_BG, border_color=BORDER,
                      text_color=TEXT_PRIMARY, placeholder_text="🔍 Search by name...").pack(side="left", padx=(0, 8))
        ctk.CTkButton(right, text="📥 Export CSV", height=36, corner_radius=8, fg_color=SUCCESS, hover_color="#00a87d",
                       font=ctk.CTkFont(size=12), command=self._export_csv).pack(side="left")

        # Table area
        self.records_scroll = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.records_scroll.pack(fill="both", expand=True, padx=24, pady=(12, 18))
        self._refresh_records()

    def _refresh_records(self):
        for w in self.records_scroll.winfo_children():
            w.destroy()
        query = self.search_var.get().strip() if hasattr(self, "search_var") else ""
        students = search_students(query) if query else get_all_students()

        if not students:
            ctk.CTkLabel(self.records_scroll, text="No records found.", font=ctk.CTkFont(size=14), text_color=TEXT_SECONDARY).pack(pady=40)
            return

        # Header row
        header = ctk.CTkFrame(self.records_scroll, fg_color=ACCENT, corner_radius=8, height=38)
        header.pack(fill="x", pady=(0, 4))
        header.pack_propagate(False)
        cols = ["ID", "Name", "CGPA", "Aptitude", "Comm", "Tech", "Intern", "Proj", "Attend%", "Certs", "Result", "Prob%", ""]
        widths = [40, 130, 50, 60, 45, 45, 45, 40, 55, 40, 70, 50, 50]
        for col, w in zip(cols, widths):
            ctk.CTkLabel(header, text=col, width=w, font=ctk.CTkFont(size=11, weight="bold"), text_color="#fff").pack(side="left", padx=2)

        for s in students:
            row = ctk.CTkFrame(self.records_scroll, fg_color=BG_CARD, corner_radius=6, height=34, border_width=1, border_color=BORDER)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)
            res_color = SUCCESS if s["prediction"] == "Placed" else DANGER
            vals = [s["id"], s["name"][:16], s["cgpa"], s["aptitude_score"], s["communication_skills"],
                    s["technical_skills"], s["internship_count"], s["project_count"],
                    s["attendance_percentage"], s["certifications_count"], s["prediction"], f'{s["probability"]}%']
            for v, w in zip(vals, widths[:-1]):
                tc = res_color if v in ("Placed", "Not Placed") else TEXT_PRIMARY
                ctk.CTkLabel(row, text=str(v), width=w, font=ctk.CTkFont(size=11), text_color=tc).pack(side="left", padx=2)
            ctk.CTkButton(row, text="🗑", width=40, height=26, corner_radius=6, fg_color="#3a1a1a", hover_color=DANGER,
                           font=ctk.CTkFont(size=11), command=lambda sid=s["id"]: self._delete_record(sid)).pack(side="left", padx=2)

    def _delete_record(self, sid):
        if messagebox.askyesno("Confirm", f"Delete record #{sid}?"):
            delete_student(sid)
            self._refresh_records()

    def _export_csv(self):
        students = get_all_students()
        if not students:
            messagebox.showinfo("Info", "No records to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                             initialfile=f"placement_report_{datetime.now().strftime('%Y%m%d')}.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=students[0].keys())
            writer.writeheader()
            writer.writerows(students)
        messagebox.showinfo("Success", f"Exported {len(students)} records to:\n{path}")

    # ══════════════════════════════════════════════════════════
    #  ANALYTICS PAGE
    # ══════════════════════════════════════════════════════════
    def show_analytics(self):
        self._clear_content()
        self._set_active_nav(3)

        scroll = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=18)

        ctk.CTkLabel(scroll, text="Analytics", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(scroll, text="Visual insights from student records", font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 16))

        stats = get_statistics()
        students = get_all_students()
        if not students:
            ctk.CTkLabel(scroll, text="No data available. Add predictions first.", font=ctk.CTkFont(size=14), text_color=TEXT_SECONDARY).pack(pady=60)
            return

        charts_row = ctk.CTkFrame(scroll, fg_color="transparent")
        charts_row.pack(fill="x", pady=(0, 14))
        charts_row.grid_columnconfigure(0, weight=1)
        charts_row.grid_columnconfigure(1, weight=1)

        # ── Pie Chart: Placement Distribution ────────────────
        pie_card = ctk.CTkFrame(charts_row, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        pie_card.grid(row=0, column=0, padx=(0, 7), sticky="nsew")
        ctk.CTkLabel(pie_card, text="  📊  Placement Distribution", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=16, pady=(12, 4))

        fig1 = Figure(figsize=(4, 3.2), facecolor=BG_CARD)
        ax1 = fig1.add_subplot(111)
        placed = stats["placed"]
        not_placed = stats["not_placed"]
        if placed + not_placed > 0:
            ax1.pie([placed, not_placed], labels=["Placed", "Not Placed"], colors=[SUCCESS, DANGER],
                    autopct="%1.1f%%", startangle=90, textprops={"color": TEXT_PRIMARY, "fontsize": 10},
                    wedgeprops={"edgecolor": BG_CARD, "linewidth": 2})
        fig1.tight_layout(pad=1)
        c1 = FigureCanvasTkAgg(fig1, pie_card)
        c1.draw()
        c1.get_tk_widget().pack(padx=10, pady=(0, 10))

        # ── Bar Chart: CGPA Distribution ─────────────────────
        bar_card = ctk.CTkFrame(charts_row, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        bar_card.grid(row=0, column=1, padx=(7, 0), sticky="nsew")
        ctk.CTkLabel(bar_card, text="  📚  CGPA Distribution", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=16, pady=(12, 4))

        cgpa_dist = get_cgpa_distribution()
        fig2 = Figure(figsize=(4, 3.2), facecolor=BG_CARD)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(BG_CARD)
        bars = ax2.bar(cgpa_dist.keys(), cgpa_dist.values(), color=["#ff6348", "#ffa502", "#6c63ff", "#2ed573", "#00cec9"], edgecolor="none", width=0.6)
        ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax2.set_ylabel("Count", color=TEXT_SECONDARY, fontsize=9)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        fig2.tight_layout(pad=1.5)
        c2 = FigureCanvasTkAgg(fig2, bar_card)
        c2.draw()
        c2.get_tk_widget().pack(padx=10, pady=(0, 10))

        # ── Skills Comparison Bar Chart ──────────────────────
        skills_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        skills_card.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(skills_card, text="  🎯  Average Skills: Placed vs Not Placed", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=16, pady=(12, 4))

        placed_students = [s for s in students if s["prediction"] == "Placed"]
        not_placed_students = [s for s in students if s["prediction"] != "Placed"]
        metrics = ["cgpa", "aptitude_score", "communication_skills", "technical_skills"]
        labels = ["CGPA", "Aptitude\n(÷10)", "Communication", "Technical"]

        def avg(lst, key):
            return np.mean([s[key] for s in lst]) if lst else 0

        placed_vals = [avg(placed_students, m) if m != "aptitude_score" else avg(placed_students, m) / 10 for m in metrics]
        np_vals = [avg(not_placed_students, m) if m != "aptitude_score" else avg(not_placed_students, m) / 10 for m in metrics]

        fig3 = Figure(figsize=(8, 3), facecolor=BG_CARD)
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor(BG_CARD)
        x = np.arange(len(labels))
        ax3.bar(x - 0.18, placed_vals, 0.35, color=SUCCESS, label="Placed", edgecolor="none")
        ax3.bar(x + 0.18, np_vals, 0.35, color=DANGER, label="Not Placed", edgecolor="none")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=9, color=TEXT_SECONDARY)
        ax3.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax3.legend(fontsize=9, facecolor=BG_CARD, edgecolor="none", labelcolor=TEXT_PRIMARY)
        for spine in ax3.spines.values():
            spine.set_visible(False)
        fig3.tight_layout(pad=1.5)
        c3 = FigureCanvasTkAgg(fig3, skills_card)
        c3.draw()
        c3.get_tk_widget().pack(padx=10, pady=(0, 10))
