#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:46:57 2024

@author: SubeenPark
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://www.reddit.com/r/nottheonion/")


for _ in range(100):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

html_content = driver.page_source
driver.quit()

urls = []
soup = BeautifulSoup(html_content, "html.parser")
for link in soup.find_all("a", class_="a", href=True):
    url = link['href']
    if url.startswith("http"):
        urls.append(url)

with open("reddit_urls.txt", "w") as file:
    for url in urls:
        file.write(url + "\n")

print("URLs have been saved to 'reddit_urls.txt' file.")



# 2nd step) Newspaper 
print(urls)
len(urls)

urls = set(urls)
len(urls)


from newspaper import Article


def save_article_contents(urls):
    data = []
    for url in urls:
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()
            
            # 기사 제목, URL, 내용을 리스트에 추가
            data.append([article.title, url, article.text])
            time.sleep(2)
        except Exception as e:
            print(f"Error processing article from {url}: {e}")
    
    # DataFrame 생성
    df = pd.DataFrame(data, columns=["Title", "URL", "Content"])
    return df

# 기사 내용을 DataFrame으로 저장
df = save_article_contents(urls)

# DataFrame 출력
df.head(10)
len(df) 
# DataFrame csv로 저장
df.to_csv("reddit_indexO.csv",index=True)




