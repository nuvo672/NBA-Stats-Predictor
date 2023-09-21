import os
import time
import pandas as pd
import requests
import tempfile
from tkinter import *
from tkinter import ttk, filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar, Style, LabelFrame, Button
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image, ImageTk
from io import BytesIO
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import threading
import json

load_dotenv()

previous_data = []

root = Tk()
root.title("NBA Player Points Prediction")
root.geometry("540x540")

style = Style()
style.theme_use("clam")

notebook = ttk.Notebook(root)
notebook.pack(fill=BOTH, expand=True)

prediction_tab = Frame(notebook)
notebook.add(prediction_tab, text="Prediction")

settings_tab = Frame(notebook)
notebook.add(settings_tab, text="Settings")

scraping_in_progress = [False]

settings = {
    "geckodriver_path": "",
    "firefox_binary_path": "",
    "csv_location": "",
}

def load_settings():
    user_home = os.path.expanduser("~")
    settings_file_path = os.path.join(user_home, "settings.json")
    try:
        with open(settings_file_path, "r") as file:
            loaded_settings = json.load(file)
            settings.update(loaded_settings)
    except FileNotFoundError:
        pass

def save_settings():
    user_home = os.path.expanduser("~")
    settings_file_path = os.path.join(user_home, "settings.json")
    with open(settings_file_path, "w") as file:
        json.dump(settings, file)

def show_settings_saved_popup():
    messagebox.showinfo("Info", "Settings saved successfully.")

def update_loading_bar(current_value):
    loading_icon["value"] = current_value
    if current_value < 100:
        loading_label["text"] = f"{current_value}%"
    else:
        loading_label["text"] = ""

    if current_value < 100:
        root.after(170, update_loading_bar, current_value + 1)

def show_loading():
    scraping_in_progress[0] = True
    loading_label.config(text="Loading...")
    loading_icon.grid(row=6, column=0, columnspan=2, pady=(20, 0))
    update_loading_bar(0)

def hide_loading():
    scraping_in_progress[0] = False
    loading_label.config(text="")
    loading_icon.grid_remove()

def update_result_display(player_name, threshold, prediction, points_list):
    prediction_value = prediction.item()
    formatted_prediction = f"{prediction_value:.2%}"
    percentage_chance_label.config(text=f"Prediction: {formatted_prediction}")
    last_10_games_points_label.config(text=f"{','.join(points_list)}")

def get_player_data(player_name, threshold):
    if scraping_in_progress[0]:
        messagebox.showinfo("Info", "Scraping is already in progress. Please wait.")
        return

    show_loading()

    def scraping_thread():
        scraping_in_progress[0] = True
        try:
            firefox_options = Options()
            firefox_options.headless = True

            with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as log_file:
                geckodriver_path = settings["geckodriver_path"]
                service = Service(executable_path=geckodriver_path, log_path=log_file.name)

                firefox_binary_path = settings["firefox_binary_path"]
                firefox_options.binary = firefox_binary_path

                driver = webdriver.Firefox(service=service, options=firefox_options)
                browser = driver

                driver.get("https://www.lineups.com/nba")

                nba_players_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "NBA Players"))
                )
                nba_players_link.click()

                search_bar = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "search-inputinput-1"))
                )
                search_bar.send_keys(player_name)
                search_bar.send_keys(Keys.RETURN)

                player_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[@class='link-black-underline']"))
                )
                player_link.click()

                time.sleep(2)
                game_logs_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/app-root/div/div/app-single-player/div/div[2]/div[2]/app-toggles-in-page-group/div/div/button[3]"))
                )
                game_logs_link.click()

                time.sleep(2)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                pts_data = soup.find_all("td", {"data-title": "PTS"})

                last_10_games_pts_data = pts_data[:10]
                pts_values = [float(pts.get_text()) for pts in last_10_games_pts_data]
                last_10_games_points_list = [str(pts) for pts in pts_values]

                time.sleep(2)

                driver.quit()

                last_10_games_pts_data = pts_data[:10]
                pts_values = [float(pts.get_text()) for pts in last_10_games_pts_data]
                boolean_values = [value > threshold for value in pts_values]

                data = {f"{player_name}_pts_values": pts_values, "greater_than_threshold": boolean_values}
                df = pd.DataFrame(data)
                file_path = os.path.join(settings["csv_location"], f"{player_name}_pts_data.csv")
                df.to_csv(file_path, index=False)

                df = pd.read_csv(file_path, names=['Points', 'Scored 30+'])

                last_10_games = df.tail(10).reset_index(drop=True)
                X = last_10_games['Points'].values.reshape(-1, 1)

                model = LogisticRegression()
                model.fit(X, last_10_games['Scored 30+'])

                next_game_prediction = model.predict_proba([[threshold]])[:, 1]

                hide_loading()

                add_to_previous_data(player_name, threshold, next_game_prediction, last_10_games_points_list)

                update_result_display(player_name, threshold, next_game_prediction, last_10_games_points_list)

        except Exception as e:
            hide_loading()
            result_label.config(text=f"An error occurred: {e}")
        finally:
            scraping_in_progress[0] = False

    scraping_thread = threading.Thread(target=scraping_thread)
    scraping_thread.start()

def add_to_previous_data(player_name, threshold, prediction, points_list):
    previous_data.append((player_name, threshold, prediction, points_list))

def display_previous_data():
    previous_data_text.config(state=NORMAL)
    previous_data_text.delete(1.0, END)
    for data in previous_data:
        player_name, threshold, prediction, points_list = data
        prediction_value = prediction.item()
        formatted_prediction = f"{prediction_value:.2%}"
        previous_data_text.insert(END, f"Player Name: {player_name}\nThreshold: {threshold}\nPrediction: {formatted_prediction}\nLast 10 Games Points: {','.join(points_list)}\n\n")
    previous_data_text.config(state=DISABLED)

def browse_geckodriver_path():
    geckodriver_path = filedialog.askopenfilename(filetypes=[("Executable files", "*.exe")])
    geckodriver_entry.delete(0, END)
    geckodriver_entry.insert(0, geckodriver_path)
    settings["geckodriver_path"] = geckodriver_path

def browse_firefox_binary_path():
    firefox_binary_path = filedialog.askopenfilename(filetypes=[("Executable files", "*.exe")])
    firefox_binary_entry.delete(0, END)
    firefox_binary_entry.insert(0, firefox_binary_path)
    settings["firefox_binary_path"] = firefox_binary_path

def browse_csv_location():
    csv_location = filedialog.askdirectory()
    csv_location_entry.delete(0, END)
    csv_location_entry.insert(0, csv_location)
    settings["csv_location"] = csv_location

def save_settings_action():
    save_settings()
    show_settings_saved_popup()

def submit_form():
    player_name = player_name_entry.get().strip()
    threshold_str = threshold_entry.get().strip()

    if not player_name:
        messagebox.showerror("Error", "Please enter a valid player name.")
        return

    try:
        threshold = float(threshold_str)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number for the threshold.")
        return

    get_player_data(player_name, threshold)

player_name_label = Label(prediction_tab, text="Player Name:")
player_name_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")

player_name_entry = ttk.Entry(prediction_tab, font=("Helvetica", 14))
player_name_entry.grid(row=0, column=1, padx=10, pady=(20, 0))

geckodriver_label = Label(settings_tab, text="GeckoDriver Path:")
geckodriver_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

geckodriver_entry = ttk.Entry(settings_tab, font=("Helvetica", 14))
geckodriver_entry.grid(row=0, column=1, padx=10, pady=10)

geckodriver_button = Button(settings_tab, text="Browse", command=browse_geckodriver_path)
geckodriver_button.grid(row=0, column=2, padx=10, pady=10)

firefox_binary_label = Label(settings_tab, text="Firefox Binary Path:")
firefox_binary_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

firefox_binary_entry = ttk.Entry(settings_tab, font=("Helvetica", 14))
firefox_binary_entry.grid(row=1, column=1, padx=10, pady=10)

firefox_binary_button = Button(settings_tab, text="Browse", command=browse_firefox_binary_path)
firefox_binary_button.grid(row=1, column=2, padx=10, pady=10)

csv_location_label = Label(settings_tab, text="CSV Files Location:")
csv_location_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

csv_location_entry = ttk.Entry(settings_tab, font=("Helvetica", 14))
csv_location_entry.grid(row=2, column=1, padx=10, pady=10)

csv_location_button = Button(settings_tab, text="Browse", command=browse_csv_location)
csv_location_button.grid(row=2, column=2, padx=10, pady=10)

save_settings_button = Button(settings_tab, text="Save Settings", command=save_settings_action)
save_settings_button.grid(row=3, column=0, columnspan=3, pady=10)

threshold_label = Label(prediction_tab, text="Threshold/Points:")
threshold_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

threshold_entry = ttk.Entry(prediction_tab, font=("Helvetica", 14))
threshold_entry.grid(row=1, column=1, padx=10, pady=10)

submit_button = Button(prediction_tab, text="Predict", command=submit_form)
submit_button.grid(row=5, column=0, columnspan=2, pady=30)

loading_label = Label(prediction_tab, text="", font=("Helvetica", 16))
loading_label.grid(row=6, column=0, columnspan=2, pady=(20, 0))

loading_icon = Progressbar(prediction_tab, mode='determinate', maximum=100)
loading_icon.grid(row=6, column=0, columnspan=2, pady=(0, 20))

result_label = Label(prediction_tab, text="", font=("Helvetica", 14))
result_label.grid(row=7, column=0, columnspan=2, pady=10)

percentage_chance_label = Label(prediction_tab, text="", font=("Helvetica", 14))
percentage_chance_label.grid(row=8, column=0, columnspan=2, pady=10)

last_10_games_label = Label(prediction_tab, text="Last 10 Games Points:", font=("Helvetica", 14))
last_10_games_label.grid(row=9, column=0, columnspan=2, pady=(20, 0))

last_10_games_points_label = Label(prediction_tab, text="", font=("Helvetica", 12))
last_10_games_points_label.grid(row=10, column=0, columnspan=2, pady=(0, 20))

previous_data_tab = Frame(notebook)
notebook.add(previous_data_tab, text="Previous Data")

previous_data_text = Text(previous_data_tab, wrap=WORD, state=DISABLED, font=("Helvetica", 12))
previous_data_text.pack(fill=BOTH, expand=True, padx=10, pady=10)

refresh_button = Button(previous_data_tab, text="Refresh Previous Data", command=display_previous_data)
refresh_button.pack(pady=10)

load_settings()

root.mainloop()
