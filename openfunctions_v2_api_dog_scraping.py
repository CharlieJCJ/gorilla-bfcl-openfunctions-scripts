"""
API Documentation Scraper

This script scrapes API documentation from apidog.com using Selenium.
It handles both root pages and subpages, saving the HTML content to organized folders.

Key features:
- Resumes from a specified index
- Logs errors for non-standard page formats
- Organizes content by API type and name
- Handles pagination and subpage navigation
"""

from selenium.webdriver.common.keys import Keys
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse

# Constants
LOG_FILE = 'log-non-traditional-formatting_Feb4_evening.txt'
TITLE_REGEX = r'<\/style>\s*<title>(.*?)<\/title>'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Scrape API documentation from apidog.com")
    parser.add_argument(
        "--resume", 
        "-r",
        help="Index to resume scraping from (default: 0)",
        type=int,
        default=0
    )
    return parser.parse_args()

def scrape_from_root(driver, url, ROOT_FOLDER):
    """
    Scrape API documentation from a root page.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL of the root page to scrape
        ROOT_FOLDER: Base folder to save scraped content
    """
    try:
        # Load page and wait for key elements
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".JsonSchemaViewer")))
    except Exception:
        # Log pages with non-standard formatting
        now = datetime.now()
        current_time = now.strftime("%b%d_%H%M%S")
        print('(ISSUE1) Non-traditional formatting:', url)
        with open(LOG_FILE, 'a') as f:    
            f.write(f"{url}  (ISSUE1) Non-traditional formatting {current_time}\n")
        return

    # Extract page content
    text_content = driver.execute_script("return document.querySelector('.JsonSchemaViewer').textContent;")
    print(text_content)
    html = driver.page_source

    # Parse API name and create folder structure
    if not re.search(TITLE_REGEX, html):
        print('(ISSUE2) Non-traditional formatting:', url)
        now = datetime.now()
        current_time = now.strftime("%b%d_%H%M%S")
        with open(LOG_FILE, 'a') as f:
            f.write(f"{url}  (ISSUE2) Non-traditional formatting {current_time}\n")
        api_link = api_name = url.replace(' ', '_').replace("/", "_")
    else:
        result = re.search(TITLE_REGEX, html).group(1).split(' - ')
        api_link = ' '.join(result[:-1])
        api_name = result[-1]

    # Clean up filenames and create folders
    api_link = api_link.replace('/', '_')
    api_link = api_link[:60] if len(api_link) > 100 else api_link
    folder_name = os.path.join(ROOT_FOLDER, api_name.replace(' ', '_'))
    
    os.makedirs(folder_name, exist_ok=True)

    # Save root page HTML
    with open(f'{folder_name}/{api_link}.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Find and process subpages
    subpages = set(re.findall(r'href="(/api-\d*)"', html))
    subpages = {urljoin(url, subpage) for subpage in subpages}
    
    for subpage_url in subpages:
        scrape_from_subpage(driver, subpage_url, folder_name, ROOT_FOLDER, api_name=api_name)

def scrape_from_subpage(driver, url, folder_name, ROOT_FOLDER, api_name):
    """
    Scrape API documentation from a subpage.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL of the subpage to scrape
        folder_name: Folder to save the subpage content
        ROOT_FOLDER: Base folder for all content
        api_name: Name of the API being scraped
    """
    try:
        # Load page and wait for key elements
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".JsonSchemaViewer")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "title")))
    except Exception:
        now = datetime.now()
        print('(ISSUE4) Non-traditional formatting:', url)
        with open(LOG_FILE, 'a') as f:    
            f.write(f"{url}  (ISSUE4) Non-traditional formatting {now.strftime('%b%d_%H%M%S')}\n")
        return

    text_content = driver.execute_script("return document.querySelector('.JsonSchemaViewer').textContent;")
    print(text_content)
    html = driver.page_source

    # Handle non-standard page formats
    if not re.search(TITLE_REGEX, html):
        print('(ISSUE3) Non-traditional formatting:', url)
        now = datetime.now()
        with open(LOG_FILE, 'a') as f:
            f.write(f"{url}  (ISSUE3) Non-traditional formatting {now.strftime('%b%d_%H%M%S')}\n")
        
        folder_name = os.path.join(ROOT_FOLDER, api_name.replace(' ', '_'))
        file_name = url.replace(' ', '_').replace("/", "_")
        
        with open(f'{folder_name}/{file_name}.html', 'w', encoding='utf-8') as f:
            f.write(html)
        return 

    # Process standard format pages
    result = re.search(TITLE_REGEX, html).group(1).split(' - ')
    api_link = ' '.join(result[:-1])
    api_link = api_link[:60] if len(api_link) > 100 else api_link
    api_link = api_link.replace('/', '_')
    
    folder_name = os.path.join(ROOT_FOLDER, api_name.replace(' ', '_'))
    
    with open(f'{folder_name}/{api_link}.html', 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    """Main execution function."""
    args = parse_args()
    print("RESUME FROM", args.resume)
    
    df = pd.read_csv('./apidog_links_cleaned.csv')
    
    for index, row in tqdm(df[args.resume:].iterrows()):
        driver = webdriver.Chrome()
        url = row["Url"]
        api_type = row["Type"]
        ROOT_FOLDER = f"data/{api_type}"
        
        scrape_from_root(driver, url, ROOT_FOLDER)
        
        # Log progress
        now = datetime.now()
        print(f"Scrape finished for Index: {index}, URL: {url}, Type: {api_type}, Time: {now.strftime('%b%d_%H%M%S')}")
        print("\n")
        
        driver.quit()

if __name__ == "__main__":
    main()