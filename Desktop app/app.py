import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd
import os

# Color scheme
DARK_BG = "#1F2937"       # Dark background
MEDIUM_BG = "#374151"     # Medium dark for frames
LIGHT_BG = "#4B5563"      # Light dark for accents
TEXT_COLOR = "#FFFFFF"    # White text
ACCENT_COLOR = "#60A5FA"  # Blue accent color
ENTRY_BG = "#374151"      # Entry background (changed from DARK_BG)
BUTTON_BG = "#60A5FA"     # Button background
BUTTON_FG = "#FFFFFF"     # Button text color
HIGHLIGHT_COLOR = "#3B82F6"  # Button active color
COMBO_BG = "#4B5563"      # Combobox background

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    return joblib.load(model_path)

def calculate_bmi(weight, feet, inches):
    height_m = ((feet * 12) + inches) * 0.0254
    return round(weight / (height_m ** 2), 2)

def predict():
    try:
        # Get input values
        age = int(age_entry.get())
        sex = sex_var.get()
        weight = float(weight_entry.get())
        feet = int(height_feet_entry.get())
        inches = int(height_inches_entry.get())
        bmi = calculate_bmi(weight, feet, inches)
        children = children_var.get()
        smoker = smoker_var.get()
        region = region_var.get()
        
        # Create derived features
        overweight_smoker = 1 if smoker == 1 and bmi > 30 else 0
        normal_nonsmoker = 1 if smoker == 0 and bmi < 30 else 0
        
        # Create region features
        region_northwest = 1 if region == "northwest" else 0
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0
        
        # Create a DataFrame with the exact same column names as training data
        input_data = pd.DataFrame([[
            age, sex, bmi, children, smoker,
            region_northwest, region_southeast, region_southwest,
            overweight_smoker, normal_nonsmoker
        ]], columns=[
            'age', 'sex', 'bmi', 'children', 'smoker',
            'region_northwest', 'region_southeast', 'region_southwest',
            'overweight_smoker', 'normal_Notsmoker'  # Note exact column name match
        ])
        
        # Load model and predict
        model = load_model()
        predicted_charge_log = model.predict(input_data)[0]
        predicted_charge = np.exp(predicted_charge_log)
        
        result_var.set(f"${predicted_charge:,.2f}")
        
    except Exception as e:
        result_var.set(f"Error: {str(e)}")

# UI Setup
root = tk.Tk()
root.title("Health Insurance Cost Predictor")
root.geometry("650x540")  
root.resizable(False, False)
root.configure(bg=DARK_BG)

# Styling
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('.', background=DARK_BG, foreground=TEXT_COLOR)
style.configure('TFrame', background=DARK_BG)
style.configure('TLabel', background=MEDIUM_BG, foreground=TEXT_COLOR, font=('Arial', 11))
style.configure('TEntry', 
                fieldbackground=ENTRY_BG, 
                foreground=TEXT_COLOR, font=('Arial', 11), 
                bordercolor=LIGHT_BG, lightcolor=LIGHT_BG, darkcolor=LIGHT_BG)

style.configure('TCombobox', 
    fieldbackground=COMBO_BG,
    foreground=TEXT_COLOR,
    background=COMBO_BG,  
    selectbackground=HIGHLIGHT_COLOR,  
    selectforeground=TEXT_COLOR,
    font=('Arial', 11),
    bordercolor=LIGHT_BG,
    lightcolor=LIGHT_BG,
    darkcolor=LIGHT_BG,
    arrowsize=15  
)


style.map('TCombobox',
    fieldbackground=[('readonly', COMBO_BG)],
    selectbackground=[('readonly', COMBO_BG)],
    background=[('readonly', COMBO_BG)],
    foreground=[('readonly', TEXT_COLOR)],
    arrowcolor=[('readonly', TEXT_COLOR)]
)

style.configure('TButton', background=BUTTON_BG, foreground=BUTTON_FG, font=('Arial', 11, 'bold'),
                borderwidth=0, focusthickness=0, focuscolor=DARK_BG)
style.map('TButton', background=[('active', HIGHLIGHT_COLOR)])
style.configure('TRadiobutton', background=MEDIUM_BG, foreground=TEXT_COLOR, font=('Arial', 11))
style.configure('TCheckbutton', background=MEDIUM_BG, foreground=TEXT_COLOR, font=('Arial', 11))
style.configure('TLabelframe', background=MEDIUM_BG, foreground=ACCENT_COLOR, font=('Arial', 12, 'bold'))
style.configure('TLabelframe.Label', background=MEDIUM_BG, foreground=ACCENT_COLOR)

# Main container
main_frame = ttk.Frame(root, padding=(20, 15))
main_frame.pack(fill=tk.BOTH, expand=True)

# Result display (top section)
result_frame = ttk.Frame(main_frame, style='TFrame')
result_frame.pack(fill=tk.X, pady=(0, 20))

result_var = tk.StringVar(value="$0.00")
result_label = ttk.Label(
    result_frame,
    textvariable=result_var,
    font=('Arial', 24, 'bold'),
    anchor="center",
    background=MEDIUM_BG,
    foreground=ACCENT_COLOR,
    padding=10
)
result_label.pack(fill=tk.X)

# Input section
input_frame = ttk.Frame(main_frame)
input_frame.pack(fill=tk.BOTH, expand=True)

# Personal Info Section
personal_frame = ttk.LabelFrame(input_frame, text="Personal Information", padding=(15, 10))
personal_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Age
ttk.Label(personal_frame, text="Age:").grid(row=0, column=0, sticky="w", pady=5)
age_entry = ttk.Entry(personal_frame, width=10)
age_entry.grid(row=0, column=1, sticky="w", pady=5, padx=5)

# Sex
ttk.Label(personal_frame, text="Sex:").grid(row=1, column=0, sticky="w", pady=5)
sex_var = tk.IntVar()
sex_frame = ttk.Frame(personal_frame, style='TFrame')
sex_frame.grid(row=1, column=1, sticky="w")
ttk.Radiobutton(sex_frame, text="Female", variable=sex_var, value=0).pack(side=tk.LEFT)
ttk.Radiobutton(sex_frame, text="Male", variable=sex_var, value=1).pack(side=tk.LEFT, padx=0)

# Children (dropdown)
ttk.Label(personal_frame, text="Children:").grid(row=2, column=0, sticky="w", pady=5)
children_var = tk.IntVar()
children_dropdown = ttk.Combobox(
    personal_frame, 
    textvariable=children_var,
    values=[0, 1, 2, 3, 4, 5],
    state="readonly",
    width=8
)
children_dropdown.grid(row=2, column=1, sticky="w", pady=5, padx=5)
children_dropdown.set(0)

# Physical Info Section
physical_frame = ttk.LabelFrame(input_frame, text="Physical Information", padding=(15, 10))
physical_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# Weight Section (top row)
weight_frame = ttk.Frame(physical_frame, style='TFrame')
weight_frame.grid(row=0, column=0, sticky="w", pady=5)

ttk.Label(weight_frame, text="Weight (kg):").pack(side=tk.LEFT, padx=(0, 0))
weight_entry = ttk.Entry(weight_frame, width=8)
weight_entry.pack(side=tk.LEFT)

# Height Section (bottom row)
height_frame = ttk.Frame(physical_frame, style='TFrame')
height_frame.grid(row=1, column=0, sticky="w", pady=5)

ttk.Label(height_frame, text="Height:").pack(side=tk.LEFT, padx=(0, 0))
height_feet_entry = ttk.Entry(height_frame, width=4)
height_feet_entry.pack(side=tk.LEFT)
ttk.Label(height_frame, text="ft").pack(side=tk.LEFT, padx=(0, 0))
height_inches_entry = ttk.Entry(height_frame, width=4)
height_inches_entry.pack(side=tk.LEFT)
ttk.Label(height_frame, text="in").pack(side=tk.LEFT)

# Lifestyle Section
lifestyle_frame = ttk.LabelFrame(input_frame, text="Lifestyle", padding=(15, 10))
lifestyle_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

# Smoker
smoker_var = tk.IntVar()
ttk.Checkbutton(
    lifestyle_frame, 
    text="Smoker", 
    variable=smoker_var
).grid(row=0, column=0, sticky="w", pady=5)

# Region
ttk.Label(lifestyle_frame, text="Region:").grid(row=1, column=0, sticky="w", pady=5)
region_var = tk.StringVar()
region_dropdown = ttk.Combobox(
    lifestyle_frame, 
    textvariable=region_var,
    values=["northeast", "northwest", "southeast", "southwest"],
    state="readonly",
    width=12
)
region_dropdown.grid(row=1, column=1, sticky="w", pady=5, padx=5)
region_dropdown.set("northeast")

# Action buttons
button_frame = ttk.Frame(input_frame, style='TFrame')
button_frame.grid(row=3, column=0, sticky="ew", pady=(15, 0))
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

calculate_btn = ttk.Button(
    button_frame, 
    text="Calculate", 
    command=predict,
    width=15
)
calculate_btn.grid(row=0, column=0, padx=5)

clear_btn = ttk.Button(
    button_frame, 
    text="Clear", 
    command=lambda: [var.set(0) for var in [sex_var, smoker_var]],
    width=15
)
clear_btn.grid(row=0, column=1, padx=5)

# Configure grid weights
input_frame.grid_columnconfigure(0, weight=1)
personal_frame.grid_columnconfigure(1, weight=1)
physical_frame.grid_columnconfigure(0, weight=1)
lifestyle_frame.grid_columnconfigure(1, weight=1)

root.mainloop()