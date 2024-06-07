#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


driver.get("https://www.theonion.com/latest")

urls = set()

for _ in range(50):
    
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    driver.execute_script("document.querySelector('button.sc-j48i5d-2.jCHulv.button.button--tertiary').click()")
    time.sleep(2)

    html_content = driver.page_source

    soup = BeautifulSoup(html_content, "html.parser")

    for link in soup.find_all("a", class_="sc-1out364-0 dPMosf js_link", href=True):
        url = link['href']
        if url.startswith("http"):
            urls.add(url)

driver.quit()

with open("theonion_urls.txt", "w") as file:
    for url in urls:
        file.write(url + "\n")

print("URLs have been saved to 'theonion_urls.txt' file.")

 
import pandas as pd
from newspaper import Article
import time

def save_article_contents(urls):
    data = []
    for url in urls:
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()
            
            
            data.append([article.title, url, article.text,article.publish_date])
            time.sleep(2)
        except Exception as e:
            print(f"Error processing article from {url}: {e}")
    
    
    df = pd.DataFrame(data, columns=["Title", "URL", "Content","Date"])
    return df


df = save_article_contents(urls)


print(df)


df.to_csv("onion_indexO.csv",index=True)














