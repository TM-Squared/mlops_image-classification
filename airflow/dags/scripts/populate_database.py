# airflow/dags/scripts/populate_db_helpers.py

import os
import pymysql
import requests


MYSQL_HOST_AIRFLOW = 'mysql'
BASE_URL = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"

def check_url_existence(url: str) -> bool | None:
    """
    Checks if a given URL exists by performing a HEAD request.

    A HEAD request is more efficient than a GET request as it only retrieves
    the headers, not the entire content, which is sufficient for checking existence.

    Args:
        url (str): The URL to check.

    Returns:
        bool | None:
            - True if the URL returns a 2xx (success) status code.
            - False if the URL returns a 4xx (client error, e.g., 404 Not Found)
              or 5xx (server error) status code.
            - None if a network error (e.g., connection refused, timeout) occurs
              during the request, meaning existence cannot be determined.
    """
    try:
        response = requests.head(url, timeout=5)
        if 200 <= response.status_code < 300:
            return True
        elif 400 <= response.status_code < 500:
            return False
        else:
            print(f"URL {url} returned an unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error checking URL {url}: {e}")
        return None

def populate_initial_metadata(label: str, num_images: int):
    """
    Inserts metadata into the 'plants_data' table and checks the existence of each image URL.

    This function connects to MySQL, iterates through image URLs for a given label,
    checks their existence online, and stores this information in the database.
    It also handles existing entries by re-checking their status if it was previously
    unknown or marked as non-existent.

    Args:
        label (str): The category/label of the images (e.g., "dandelion", "grass").
        num_images (int): The total number of images expected for this label (0-indexed).
    """
    db_user = os.getenv("MYSQL_USER")
    db_password = os.getenv("MYSQL_PASSWORD")
    db_database = os.getenv("MYSQL_DATABASE")

    if not all([db_user, db_password, db_database]):
        raise ValueError("Missing MySQL database connection environment variables (MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE).")

    try:
        connection = pymysql.connect(
            host=MYSQL_HOST_AIRFLOW,
            user=db_user,
            password=db_password,
            database=db_database,
            cursorclass=pymysql.cursors.DictCursor
        )
        print("Successfully connected to MySQL for metadata population!")
    except pymysql.MySQLError as e:
        raise ConnectionError(f"Error connecting to MySQL: {e}") from e

    try:
        print(f"\nPopulating data for label: {label}")
        with connection.cursor() as cursor:
            for i in range(num_images):
                image_name = f"{i:08d}.jpg"
                url = f"{BASE_URL}/{label}/{image_name}"

                cursor.execute("SELECT id, image_exists FROM plants_data WHERE url_source = %s", (url,))
                result = cursor.fetchone()

                if not result:
                    exists = check_url_existence(url)
                    sql = "INSERT INTO `plants_data` (`url_source`, `label`, `image_exists`) VALUES (%s, %s, %s)"
                    cursor.execute(sql, (url, label, exists))
                    status_msg = "Exists" if exists is True else ("Does Not Exist" if exists is False else "Verification Error")
                    print(f"Inserted: {url} ({status_msg})")
                else:
                    existing_status = result['image_exists']
                    if existing_status is None or existing_status is False:
                        exists = check_url_existence(url)
                        if exists is not None and exists != existing_status:
                            sql = "UPDATE `plants_data` SET `image_exists` = %s WHERE `id` = %s"
                            cursor.execute(sql, (exists, result['id']))
                            status_msg = "Updated: Exists" if exists is True else "Updated: Does Not Exist"
                            print(f"{status_msg}: {url}")
                        else:
                            status_msg = "Already exists" if existing_status is True else "Already does not exist" if existing_status is False else "Status unchanged or verification error"
                            print(f"{status_msg}: {url}")
                    else:
                        print(f"Already exists (status {existing_status}): {url}")
        connection.commit()
    finally:
        connection.close()
        print("MySQL connection closed after metadata population.")