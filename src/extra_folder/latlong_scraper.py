import requests
from bs4 import BeautifulSoup
import pandas as pd
BASE_URL = "https://www.google.com/search?q="
# Headers to simulate a real browser visit
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"
}

# Function to scrape latitude and longitude
def get_codrdinates(sector):
    search_term = f"{sector} ahmedabad longitude & latitude"
    response = requests.get(BASE_URL + search_term, headers=HEADERS)
    print(response)
    if response.status_code == 200:
          soup = BeautifulSoup(response.content, 'html.parser')
          coordinates_div = soup.find("div", class_="Z0LcW t2b5Cf")
          print(coordinates_div)
          if coordinates_div:
              print(coordinates_div.text)
              return coordinates_div.text
    return None
data = pd.read_csv('streamlit/artifacts/sector.csv')
areas = data['sector'].tolist()
#Create a DataFrame
data2 = []

# Iterate over sectors and fetch coordinates
for sector in areas:
    coordinates = get_codrdinates(sector)  # Corrected the typo in 'get_coordinates'
    data2.append({"sector": sector, "coordinates": coordinates})

# Create a DataFrame from the list of data
df = pd.DataFrame(data2)

# Save DataFrame to a CSV file
df.to_csv("data/latlong.csv", index=False)