#!/usr/bin/env python
# coding: utf-8

# In[20]:


import selenium 
import sys
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.support import expected_conditions as EC 
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException,TimeoutException 
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time
import re
import csv
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


# In[14]:


def login():

    button_login_window = driver.find_element(By.CSS_SELECTOR, '#mweb-unauth-container > div > div > div.QLY._he.zI7.iyn.Hsu > div > div.Eqh.KS5.hs0.un8.C9i.TB_ > div.wc1.zI7.iyn.Hsu > button > div > div')
    button_login_window.click()

    input_email = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div[2]/div/div/div/div/div/div[4]/form/div[2]/fieldset/span/div/input')
    input_email.send_keys("Your email")
    input_password = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div[2]/div/div/div/div/div/div[4]/form/div[4]/fieldset/span/div/input')
    input_password.send_keys('Your Password')

    button_login = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div[2]/div/div/div/div/div/div[4]/form/div[7]/button')
    button_login.click()

    time.sleep(20)
    driver.implicitly_wait(10)


# In[28]:


search_word= '캘리그라피'
target_num_images = 1000

options = Options()
options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.implicitly_wait(3)

# url에 접근한다.
driver.get('https://www.pinterest.com/')
driver.implicitly_wait(3)
time.sleep(5)


# 경로와 검색어 설정
imagedir = 'images'
url_file_path = os.path.join(imagedir, "URLs.txt")  # URL을 저장할 파일 경로

login()
# 검색어 입력하기
input_search_keyword = driver.find_element(By.CSS_SELECTOR,'#searchBoxContainer > div > div > div.ujU.zI7.iyn.Hsu > input[type=text]')
for _ in range(100):
    input_search_keyword.send_keys(Keys.BACKSPACE)
    time.sleep(0.01)
input_search_keyword.send_keys('캘리그라피')
input_search_keyword.send_keys(Keys.ENTER)

driver.implicitly_wait(5)

print('Searching word (' + search_word + ')')



try:
    os.makedirs(imagedir, exist_ok=True)
    os.makedirs(os.path.join(imagedir, search_word), exist_ok=True)
    with open(url_file_path, "w") as f:  # URL 파일 초기화
        pass
except Exception as e:
    print(f"Directory creation failed: {e}")

links_saved = set()
force_scroll_down = 0
target_num_images = 1000  # 목표 이미지 수

# 이미지와 URL 크롤링
while len(links_saved) < target_num_images:
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, "//img[@srcset]"))
    )
    count_new_images = 0
    
    for image in images:
        time.sleep(0.5)
        links = image.get_property('srcset')
        time.sleep(0.5)
        url = list(filter(lambda x: x.startswith('https'), links.split(' ')))[-1]
        if url not in links_saved:
            links_saved.add(url)
            count_new_images += 1
            print(f'Image URL: {url}')
            imagename = url[url.rfind('/')+1:]
            response = requests.get(url)
            file_path = os.path.join(imagedir, search_word, imagename)
            with open(file_path, "wb") as file:
                file.write(response.content)
            with open(url_file_path, "a") as file:
                file.write(url + "\n")
    
    print(str(len(links_saved)) + ' / ' + str(target_num_images))
    
    if count_new_images == 0:
        if force_scroll_down < 800:  # 스크롤 다운 횟수 증가
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            force_scroll_down += 1
            time.sleep(5)  # 로드 대기 시간 증가
            driver.implicitly_wait(4)
        else:
            print('Force break due to no new images')
            break
    elif len(images) > 0:
        action = ActionChains(driver)
        action.move_to_element(images[-1]).perform()
        force_scroll_down = 0
    else:
        break

print(f'Total images saved: {len(links_saved)}')


# In[ ]:





# In[ ]:


def find_and_save_images():
    search_word = '캘리그라피'
    imagedir = 'images'
    url_file_path = os.path.join(imagedir, "URLs.txt")
    os.makedirs(imagedir, exist_ok=True)
    os.makedirs(os.path.join(imagedir, search_word), exist_ok=True)
    with open(url_file_path, "w") as f:
        pass
    input_search_keyword = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,'#searchBoxContainer > div > div > div.ujU.zI7.iyn.Hsu > input[type=text]'))
    )
    input_search_keyword.send_keys(search_word + Keys.ENTER)
    links_saved = set()
    target_num_images = 1000

    while len(links_saved) < target_num_images:
        images = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//img[@srcset]"))
        )
        for img in images:
            try:
                srcset = img.get_attribute('srcset')
                url = list(filter(lambda x: x.startswith('https'), srcset.split(' ')))[-1]
                if url not in links_saved:
                    links_saved.add(url)
                    print(f'Image URL: {url}')
                    imagename = url[url.rfind('/')+1:]
                    response = requests.get(url)
                    file_path = os.path.join(imagedir, search_word, imagename)
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                    with open(url_file_path, "a") as file:
                        file.write(url + "\n")
            except Exception as e:
                continue
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(5)

    print(f'Total images saved: {len(links_saved)}')


