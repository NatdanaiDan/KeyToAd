from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
import pandas as pd
import multiprocessing

chrome_options = Options()

# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')

def scrape_page(i):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://www.watsons.co.th/th/search?text=%E0%B9%80%E0%B8%84%E0%B8%A3%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%87%E0%B8%AA%E0%B8%B3%E0%B8%AD%E0%B8%B2%E0%B8%87&useDefaultSearch=false&sortCode=bestSeller&pageSize=64&currentPage={i}")
    time.sleep(5)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    div_element = soup.find('div', class_='cx-product-container').find_all('a', class_="ClickSearchResultEvent_Class gtmAlink")
    all_link = []
    for link in div_element:
        text = "https://www.watsons.co.th/" + str(link.get('href'))
        all_link.append(text)
    all_link_inpage = list(dict.fromkeys(all_link))
    lst_name = []
    lst_detail = []
    print(f'len {len(all_link_inpage)} Page {i}')
    pbar = tqdm(all_link_inpage)
    for i in pbar:
        driver.get(i)
        time.sleep(3)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Keep all link in page
        div_name = soup.find('div', class_='product-name')
        div_detail = soup.find('div', class_='product-details')
        lst_name.append(div_name.text)
        lst_detail.append(div_detail.text)
    driver.quit()
    return lst_name, lst_detail

if __name__ == "__main__":
    num_pages = 93
    num_processes = multiprocessing.cpu_count()  # You can adjust this as needed

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(scrape_page, range(num_pages))
    pool.close()
    pool.join()

    all_names = []
    all_details = []

    for names, details in results:
        all_names.extend(names)
        all_details.extend(details)

    df = pd.DataFrame(list(zip(all_names, all_details)), columns=['Name', 'Detail'])
    df.to_csv("Watson64.csv", index=False)
