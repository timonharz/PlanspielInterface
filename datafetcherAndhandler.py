from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import schedule

driver = webdriver.Chrome()
driver.implicitly_wait(0.5)

username = ""
password = ""
#global variables
loggedIn = False
totalCapitalInt = 0
print(__name__)

def loginAndFetch(): 
    driver.get("https://trading.planspiel-boerse.de/web/auth/login")
    driver.implicitly_wait(3)
    usernameField = driver.find_element(By.XPATH, "/html/body/app-root/app-authentication/div/div/div[2]/app-login/div/form/div[1]/input")
    usernameField.send_keys(username)
    usernameField.send_keys(Keys.RETURN)
    passwordField = driver.find_element(By.CSS_SELECTOR, "#password")
    passwordField.send_keys(password)
    passwordField.send_keys(Keys.RETURN)
    loginButton = driver.find_element(By.XPATH, "/html/body/app-root/app-authentication/div/div/div[2]/app-login/div/form/div[4]/button")
    loginButton.click()
    driver.implicitly_wait(7)
   # loggedIn = True
    depotButton = driver.find_element(By.XPATH, "/html/body/app-root/app-home/app-dashboard/app-main/div[1]/div/div/div/div[2]/button")
    depotButton.click()
    driver.implicitly_wait(5)
    totalCapital = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".flex-lg-row > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > p:nth-child(1)"))).get_attribute("textContent")
    print(totalCapital)
    print(type(totalCapital))
    totalCapitalString = totalCapital.replace(" ", "").replace(".", "").replace(",", "").replace("€", "")
    totalCapitalString = totalCapitalString[:-2]
    print(totalCapitalString)
    totalCapitalInt = int(totalCapitalString)

def readCapital():
    totalCapital = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".flex-lg-row > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > p:nth-child(1)"))).get_attribute("textContent")
    print(totalCapital)
    print(type(totalCapital))
    totalCapitalString = totalCapital.replace(" ", "").replace(".", "").replace(",", "").replace("€", "")
    totalCapitalString = totalCapitalString[:-2]
    print(totalCapitalString)
    totalCapitalInt = int(totalCapitalString)
   
loginAndFetch()
#schedule.every(1).minutes.do(readCapital)

#while True:
 #   schedule.run_pending()
  #  time.sleep(1)
