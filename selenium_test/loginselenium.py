from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager  # Use EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import easygui

# Initialize EdgeDriver using webdriver-manager
driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))

# Open the login page
driver.get("http://127.0.0.1:8000/login/")
print("Opened the login page.")

# Maximize the browser window
driver.maximize_window()

# Wait for the page to load completely
time.sleep(3)

# Wait for the email field to be visible and interact with it
wait = WebDriverWait(driver, 10)

# Step 1: Log in with valid credentials
email_field = wait.until(EC.visibility_of_element_located((By.NAME, "email")))
email_field.clear()
email_field.send_keys('admin@gmail.com')  # Use the appropriate email
print("Entered email.")
time.sleep(1)

password_field = wait.until(EC.visibility_of_element_located((By.NAME, 'password')))
password_field.clear()
password_field.send_keys('admin')  # Use the appropriate password
print("Entered password.")
time.sleep(1)

# Locate and click the login button
login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
print("Login button found. Clicking the button.")
login_button.click()

# Wait for the page to process login
time.sleep(3)

# Check for login success or failure
try:
    # Check if there's an error message displayed
    error_message = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.error-message')))  # Adjust the selector based on your error message element
    if error_message.is_displayed():
        easygui.msgbox("Login Failed: " + error_message.text)
        print("Login failed. Error message displayed.")
except Exception as e:
    # If no error message is found, assume login was successful
    easygui.msgbox("Add Category successful.")
    print("Add Category successful.")

# Step 2: Log in with invalid credentials
# Refresh the page to reset the state
driver.get("http://127.0.0.1:8000/login/")
print("Refreshed the login page.")

# Wait for the page to load completely again
time.sleep(3)

# Enter invalid email and password
invalid_email_field = wait.until(EC.visibility_of_element_located((By.NAME, "email")))
invalid_email_field.clear()
invalid_email_field.send_keys('invalid_user@gmail.com')  # Use an email that does not exist
print("Entered invalid email.")
time.sleep(1)

invalid_password_field = wait.until(EC.visibility_of_element_located((By.NAME, 'password')))
invalid_password_field.clear()
invalid_password_field.send_keys('wrongpassword')  # Use an incorrect password
print("Entered invalid password.")
time.sleep(1)

# Locate and click the login button again
invalid_login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
print("Invalid login button found. Clicking the button.")
invalid_login_button.click()

# Wait for the page to process login
time.sleep(3)

# Check for login failure
try:
    # Check if there's an error message displayed
    error_message = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.error-message')))  # Adjust the selector based on your error message element
    if error_message.is_displayed():
        easygui.msgbox("Login Failed: " + error_message.text)
        print("Deleted Category")
except Exception as e:
    easygui.msgbox("Deleted Category")
    print("Unexpected behavior: Login was successful with invalid credentials.")

# Close the browser
driver.quit()
