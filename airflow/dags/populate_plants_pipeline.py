from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
from io import BytesIO
import requests 

from scripts.populate_database import populate_initial_metadata

S3_BUCKET_NAME = 'raw-data'

def _download_and_upload_to_s3(**kwargs):
    """
    Downloads images from source URLs and uploads them to MinIO/S3.

    This function pulls a list of image records (ID, source URL, label) from XCom,
    downloads each image, uploads it to the configured MinIO/S3 bucket, and then
    updates the MySQL database with the S3 URL for the uploaded image.

    Args:
        **kwargs: Airflow context keywords, including 'ti' (TaskInstance) for XCom.
    """
    ti = kwargs['ti']
    json_data = ti.xcom_pull(task_ids='get_new_data_for_s3_upload')

    if not json_data:
        print("No new data to process for S3 upload. Exiting.")
        return

    df = pd.read_json(json_data)

    s3_hook = S3Hook(aws_conn_id='s3_connec')
    
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')

    if not s3_hook.check_for_bucket(S3_BUCKET_NAME):
        print(f"S3 bucket '{S3_BUCKET_NAME}' does not exist. Attempting to create it.")
        try:
            s3_hook.get_conn().create_bucket(Bucket=S3_BUCKET_NAME)
            print(f"S3 bucket '{S3_BUCKET_NAME}' created successfully.")
        except Exception as e:
            print(f"Failed to create S3 bucket '{S3_BUCKET_NAME}': {e}. Please create it manually if this error persists.")

    for index, row in df.iterrows():
        url_source = row['url_source']
        db_id = row['id']
        label = row['label']

        try:
            print(f"Downloading image from: {url_source}")
            response = requests.get(url_source, stream=True, timeout=10)
            response.raise_for_status()

            file_name = url_source.split('/')[-1]
            s3_key = f"raw/{label}/{file_name}"
            print(f"Uploading {file_name} to S3 key: {s3_key}")

            s3_hook.load_file_obj(
                file_obj=BytesIO(response.content),
                key=s3_key,
                bucket_name=S3_BUCKET_NAME,
                replace=True
            )
            print(f"Image '{s3_key}' uploaded successfully to MinIO.")

            s3_url_in_db = f"s3://{S3_BUCKET_NAME}/{s3_key}"
            
            mysql_hook.run(
                "UPDATE plants_data SET url_s3 = %s WHERE id = %s",
                parameters=(s3_url_in_db, db_id)
            )
            print(f"MySQL DB updated for ID {db_id} with S3 URL: {s3_url_in_db}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download or connect for {url_source}: {e}. Skipping upload.")
        except Exception as e:
            print(f"An unexpected error occurred processing {url_source} for S3 upload: {e}. Skipping upload.")

with DAG(
    dag_id='plants_data_ingestion_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['plants', 'database', 'metadata', 's3', 'minio'],
    doc_md="""
    

    This DAG orchestrates the ingestion of plant image metadata and the download
    and upload of actual images to MinIO/S3.

    1.  **Populate Metadata:** Inserts initial image URLs and checks their existence.
    2.  **Get Data for S3 Upload:** Queries the DB for valid image URLs that haven't
        been uploaded to S3 yet.
    3.  **Download & Upload:** Downloads images and uploads them to MinIO/S3,
        then updates the database with the S3 paths.
    """
) as dag:
    populate_dandelion_task = PythonOperator(
        task_id='populate_dandelion_metadata',
        python_callable=populate_initial_metadata,
        op_kwargs={'label': 'dandelion', 'num_images': 200},
    )

    populate_grass_task = PythonOperator(
        task_id='populate_grass_metadata',
        python_callable=populate_initial_metadata,
        op_kwargs={'label': 'grass', 'num_images': 200},
    )

    get_new_data_for_s3_upload = PythonOperator(
        task_id='get_new_data_for_s3_upload',
        
        python_callable=lambda: MySqlHook(mysql_conn_id='mysql_default').get_pandas_df(
            sql="SELECT id, url_source, label FROM plants_data WHERE image_exists = TRUE AND url_s3 IS NULL;"
        ).to_json(orient='records'),
        provide_context=True
    )

    download_and_upload_task = PythonOperator(
        task_id='download_and_upload_images_to_s3',
        python_callable=_download_and_upload_to_s3,
        provide_context=True,
    )

    [populate_dandelion_task, populate_grass_task] >> get_new_data_for_s3_upload >> download_and_upload_task