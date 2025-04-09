from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import easygui

# Initialize EdgeDriver with SSL bypass options
options = webdriver.EdgeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--allow-insecure-localhost')  # Allow insecure localhost connections
driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)

# Open the login page
driver.get("http://127.0.0.1:8000/login/")
driver.maximize_window()

# Wait for the page to load completely
time.sleep(3)

# Log in with valid credentials
wait = WebDriverWait(driver, 10)
email_field = wait.until(EC.visibility_of_element_located((By.NAME, "email")))
email_field.clear()
email_field.send_keys('admin@gmail.com')
time.sleep(1)

password_field = wait.until(EC.visibility_of_element_located((By.NAME, 'password')))
password_field.clear()
password_field.send_keys('admin')
time.sleep(1)

login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
login_button.click()

# Wait for the page to process login
time.sleep(3)

# Step 1: Navigate to the Add Category page
try:
    driver.get("http://127.0.0.1:8000/add-category/")
    print("Navigated to the Add Category page.")
except Exception as e:
    easygui.msgbox("Failed to load the Add Category page.")
    print("Error loading Add Category page:", e)
    driver.quit()

# Step 2: Fill in the Add Category form
try:
    category_name_field = wait.until(EC.visibility_of_element_located((By.NAME, "category_name")))
    category_name_field.clear()
    category_name_field.send_keys('New Category')
    print("Entered category name.")
    time.sleep(1)

    # If there is a description field, fill it
    try:
        description_field = driver.find_element(By.NAME, 'description')  # Replace 'description' with the actual field name
        description_field.clear()
        description_field.send_keys('This is a test category description.')
        print("Entered category description.")
        time.sleep(1)
    except Exception as e:
        print("Description field not found:", e)

    # Click the submit button
    submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
    submit_button.click()
    print("Clicked the submit button to add the category.")

    # Wait for processing and then check for the success message
    time.sleep(3)

    # Check for success message
    try:
        success_message = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.alert.alert-success')))  # Adjust selector if needed
        if success_message.is_displayed():
            easygui.msgbox("Category added successfully!")
            print("Category added successfully.")
    except Exception as e:
        print("Success message not found:", e)
        print("Page source at the time of failure:", driver.page_source)

except Exception as e:
    print("Error while trying to add a category:", e)
    print("Page source at the time of error:", driver.page_source)

# Close the browser
driver.quit()
