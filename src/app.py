import tkinter as tk
from tkinter import ttk, messagebox
import threading
import paths as PATH

from evaluation_with_visual import run_evaluation_USERS, run_evaluation_IMPOSTORS, plot_biometric_metrics

class BiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👁️ System Rozpoznawania Tęczówki")
        self.root.geometry("420x540")
        self.root.resizable(False, False)
        self.tp = self.fn = self.tn = self.fp = 0
        self.users_limit_var = tk.IntVar(value=100)

        self.setup_ui()

    def setup_ui(self):
        frame_settings = ttk.LabelFrame(self.root, text=" Ustawienia ", padding=10)
        frame_settings.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_settings, text="Liczba użytkowników do testu:").pack(anchor="w")
        
        slider_frame = ttk.Frame(frame_settings)
        slider_frame.pack(fill="x", pady=5)
        
        slider = ttk.Scale(slider_frame, from_=5, to=100, variable=self.users_limit_var, orient="horizontal")
        slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.lbl_limit = ttk.Label(slider_frame, text="100")
        self.lbl_limit.pack(side="right")
        self.users_limit_var.trace_add("write", lambda *args: self.lbl_limit.config(text=str(self.users_limit_var.get())))

        frame_actions = ttk.LabelFrame(self.root, text=" Uruchom Testy ", padding=10)
        frame_actions.pack(fill="x", padx=10, pady=5)

        self.btn_users = ttk.Button(frame_actions, text="1. Test Legalnych Użytkowników", command=self.start_users_eval)
        self.btn_users.pack(fill="x", pady=2)

        self.btn_impostors = ttk.Button(frame_actions, text="2. Test Impostorów (Włamania)", command=self.start_impostors_eval)
        self.btn_impostors.pack(fill="x", pady=2)

        self.btn_plots = ttk.Button(frame_actions, text="3. Generuj Wykresy Wyników", command=self.show_plots)
        self.btn_plots.pack(fill="x", pady=10)

        frame_results = ttk.LabelFrame(self.root, text=" Bieżące Wyniki ", padding=10)
        frame_results.pack(fill="x", padx=10, pady=5)

        self.lbl_tp_fn = ttk.Label(frame_results, text="TP: 0  |  FN: 0", font=("Helvetica", 10, "bold"))
        self.lbl_tp_fn.pack(anchor="w", pady=2)

        self.lbl_tn_fp = ttk.Label(frame_results, text="TN: 0  |  FP: 0", font=("Helvetica", 10, "bold"))
        self.lbl_tn_fp.pack(anchor="w", pady=2)
        
        self.lbl_accuracy = ttk.Label(frame_results, text="Skuteczność (Accuracy): N/A", font=("Helvetica", 10, "bold"), foreground="#2563EB")
        self.lbl_accuracy.pack(anchor="w", pady=4)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", padx=15, pady=(10, 0))

        self.lbl_status = ttk.Label(self.root, text="Status: Gotowy do pracy.", foreground="green")
        self.lbl_status.pack(side="bottom", anchor="w", padx=10, pady=10)

    def set_loading_state(self, message):
        self.lbl_status.config(text=message, foreground="blue")
        self.btn_users.state(['disabled'])
        self.btn_impostors.state(['disabled'])
        self.btn_plots.state(['disabled'])
        self.progress.start(10) 

    def set_ready_state(self):
        self.lbl_status.config(text="Status: Gotowy.", foreground="green")
        self.btn_users.state(['!disabled'])
        self.btn_impostors.state(['!disabled'])
        self.btn_plots.state(['!disabled'])
        self.progress.stop() 
        self.update_results_labels()

    def update_results_labels(self):
        self.lbl_tp_fn.config(text=f"TP (Poprawne akceptacje): {self.tp}  |  FN (Błędne odrzucenia): {self.fn}")
        self.lbl_tn_fp.config(text=f"TN (Poprawne odrzucenia): {self.tn}  |  FP (Włamania): {self.fp}")
        
        total = self.tp + self.fn + self.tn + self.fp
        if total > 0:
            accuracy = ((self.tp + self.tn) / total) * 100
            self.lbl_accuracy.config(text=f"Skuteczność (Accuracy): {accuracy:.2f}%")
        else:
            self.lbl_accuracy.config(text="Skuteczność (Accuracy): N/A")

    def start_users_eval(self):
        self.set_loading_state("Status: Przetwarzanie użytkowników... (to może potrwać)")
        limit = self.users_limit_var.get()
        
        def task():
            try:
                self.tp, self.fn, _ = run_evaluation_USERS(PATH.DATA_DIR, limit)
            except Exception as e:
                messagebox.showerror("Błąd", f"Wystąpił błąd:\n{e}")
            finally:
                self.root.after(0, self.set_ready_state)
                
        threading.Thread(target=task, daemon=True).start()

    def start_impostors_eval(self):
        self.set_loading_state("Status: Testowanie impostorów... (to może potrwać)")
        limit = self.users_limit_var.get()
        
        def task():
            try:
                self.tn, self.fp, _ = run_evaluation_IMPOSTORS(PATH.DATA_DIR, limit)
            except Exception as e:
                messagebox.showerror("Błąd", f"Wystąpił błąd:\n{e}")
            finally:
                self.root.after(0, self.set_ready_state)
                
        threading.Thread(target=task, daemon=True).start()

    def show_plots(self):
        total = self.tp + self.fn + self.tn + self.fp
        if total == 0:
            messagebox.showwarning("Brak danych", "Uruchom najpierw ewaluację, aby wygenerować wykresy!")
            return
        
        plot_biometric_metrics(self.tp, self.fn, self.tn, self.fp)

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = BiometricApp(root)
    root.mainloop()