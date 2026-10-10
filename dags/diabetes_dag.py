from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline_tasks import extract_data, preprocess_data
from src.train import train_and_save

DATA_DIR = "/opt/airflow/project/data/processed"
RAW_FILE = f"{DATA_DIR}/diabetes_clustered.csv"
TRAIN_FILE = f"{DATA_DIR}/diabetes_training.csv"

with DAG(
    dag_id="diabetes_retraining",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    task_extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
        op_kwargs={"path": RAW_FILE},
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_kwargs={"path": RAW_FILE, "output_path": TRAIN_FILE},
    )

    task_retrain = PythonOperator(
        task_id="retrain_and_register",
        python_callable=train_and_save,
        op_kwargs={"data_path": TRAIN_FILE},
    )

    task_extract >> task_preprocess >> task_retrain