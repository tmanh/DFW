# export_map_selenium.py
import os, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

HTML = "train_test_station_map.html"
OUT  = "train_test_station_map.png"

# If your tiles don't load with file://, use the http.server method (see below).
url = "file://" + os.path.abspath(HTML)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-gpu")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=2200,1400")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=opts)

driver.get(url)
time.sleep(6)  # let leaflet tiles load
driver.save_screenshot(OUT)
driver.quit()

print("Saved:", OUT)
