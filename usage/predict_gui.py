import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load trained model
# -----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pm25.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM", "PM2.5_lag_1h","PM2.5_lag_3h","PM2.5_lag_6h","PM2.5_lag_12h","PM2.5_lag_24h","PM2.5_roll_mean_3h","PM2.5_roll_std_3h", "PM2.5_roll_mean_6h","PM2.5_roll_std_6h",
    "PM2.5_roll_mean_12h","PM2.5_roll_std_12h","PM2.5_roll_mean_24h","PM2.5_roll_std_24h", "hour","dayofweek","month", "hour_sin","hour_cos","month_sin","month_cos", "wd_ENE","wd_ESE","wd_N","wd_NE","wd_NNE","wd_NNW","wd_NW",
    "wd_S","wd_SE","wd_SSE","wd_SSW","wd_SW","wd_W","wd_WNW","wd_WSW"
]

uploaded_df = None
prediction_df = None

# Forecast logic (5 da ys)
# -----------------------
def forecast_next_5_days(df):
    hours = 120
    last_row = df.iloc[-1].copy()
    preds, times = [], []

    current = last_row.copy()
    start_time = df.index[-1]

    for i in range(hours):
        ts = start_time + pd.Timedelta(hours=i + 1)

        current["hour"] = ts.hour
        current["dayofweek"] = ts.dayofweek
        current["month"] = ts.month
        current["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        current["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        current["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        current["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

        pred = model.predict(pd.DataFrame([current]))[0]
        preds.append(pred)
        times.append(ts)

        current["PM2.5_lag_1h"] = pred

    hourly = pd.DataFrame(
        {"datetime": times, "PM2.5": preds}
    ).set_index("datetime")

    return hourly.resample("D").mean()

# Button actions
#-------------
def upload_csv():
    global uploaded_df
    path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not path:
        return

    uploaded_df = pd.read_csv(path, index_col=0, parse_dates=True)
    status_label.config(text="CSV loaded successfully", fg="green")
    output_text.delete("1.0", tk.END)

def predict():
    global prediction_df
    if uploaded_df is None:
        messagebox.showerror("Error", "Please upload a CSV first.")
        return

    try:
        X = uploaded_df[FEATURE_COLUMNS]
        prediction_df = forecast_next_5_days(X)

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, prediction_df.round(2).to_string())

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

def export_prediction():
    if prediction_df is None:
        messagebox.showerror("Error", "No prediction to export.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    if save_path:
        prediction_df.to_csv(save_path)
        messagebox.showinfo("Success", "Prediction exported successfully.")

# GUI Layout
# -----------
root = tk.Tk()
root.title("Air Quality PM2.5 Prediction Tool")

# Center window
W, H = 520, 320
x = (root.winfo_screenwidth() - W) // 2
y = (root.winfo_screenheight() - H) // 2
root.geometry(f"{W}x{H}+{x}+{y}")
root.resizable(False, False)

# Title
# -------
tk.Label(
    root,
    text="Air Quality PM2.5 Prediction Tool",
    font=("Segoe UI", 15, "bold")
).pack(pady=(10, 2))

tk.Label(
    root,
    text="Random Forest — Next 5 Days Forecast",
    font=("Segoe UI", 10),
    fg="gray"
).pack(pady=(0, 10))

# Main Layout
# ----------
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=15, pady=10)

# Left panel
left_frame = tk.Frame(main_frame, width=200)
left_frame.pack(side="left", fill="y", padx=(0, 15))

btn_style = {
    "width": 18,
    "height": 2,
    "font": ("Segoe UI", 10)
}

tk.Button(left_frame, text="Upload CSV", command=upload_csv, **btn_style).pack(pady=6)
tk.Button(left_frame, text="Predict", command=predict, **btn_style).pack(pady=6)
tk.Button(left_frame, text="Export Prediction", command=export_prediction, **btn_style).pack(pady=6)

status_label = tk.Label(left_frame, text="", font=("Segoe UI", 9))
status_label.pack(pady=10)

# Right panel
right_frame = tk.Frame(main_frame, bg="#f2f2f2", bd=1, relief="solid")
right_frame.pack(side="right", fill="both", expand=True)

output_text = tk.Text(
    right_frame,
    font=("Consolas", 16),
    bg="#fafafa",
    relief="flat",
    wrap="none"
)
output_text.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()



