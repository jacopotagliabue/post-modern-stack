"""

    One-off script to upload the Coveo Shopping Dataset into a Snowflake warehouse.

    We leverage here:

    * the dataset wrapper provided by RecList (https://github.com/jacopotagliabue/reclist)
    * the data logic explained in our paas ingestion repo (https://github.com/jacopotagliabue/paas-data-ingestion)

    The full dataset in the original shape can be found in the SIGIR data challenge repository (https://github.com/coveooss/SIGIR-ecom-data-challenge):
    please star the repos involved and cite the relevant paper if you find this project useful for your work.

"""
import os
import csv
import time
import uuid
import json
from dotenv import load_dotenv
from datetime import datetime
from reclist.datasets import CoveoDataset
import tempfile
import sys
sys.path.insert(0, '..')
from src.clients.snowflake_client import SnowflakeClient
# load envs
load_dotenv(verbose=True, dotenv_path='../src/.env')


def create_schema(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    ):
    sql_query = "create schema IF NOT EXISTS {}.{};".format(snowflake_db.upper(),
                                                            snowflake_schema.upper())

    return snowflake_client.execute_query(sql_query)


def create_table(
    snowflake_client : SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_table: str
    ):
    """
    
    Original file is from https://github.com/coveooss/SIGIR-ecom-data-challenge
    but we are uploading a slightly pre-processed version of the events included in
    the RecList package (https://github.com/jacopotagliabue/reclist)
    
    """
    # attention, table is RE-CREATED!
    print("====== TABLE IS BEING DROPPED AND RE-CREATED =====")
    sql_query = """
    CREATE OR REPLACE TABLE 
    {}.{}.{}(
        etl_timestamp int,
        etl_id VARCHAR(36),
        event_type string,
        api_key VARCHAR(36),
        event_date DATE,
        raw_data VARIANT
    );
    """.format(snowflake_db.upper(),
               snowflake_schema.upper(),
               snowflake_table.upper())

    return snowflake_client.execute_query(sql_query, is_debug=True)


def use_database(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str
    ):
    sql_query = "USE DATABASE {};".format(snowflake_db.upper())

    return snowflake_client.execute_query(sql_query)


def stage_data(
    snowflake_client: SnowflakeClient, 
    snowflake_schema: str,
    snowflake_table: str,
    data_file: str,
    data_folder: str
    ):
    sql_query = "PUT file://{}/{} @{}.%{} auto_compress=true overwrite=true".format(
        data_folder,
        data_file,
        snowflake_schema.upper(),
        snowflake_table.upper()
        )

    return snowflake_client.execute_query(sql_query, is_debug=True)


def copy_data(
    snowflake_client: SnowflakeClient, 
    snowflake_schema: str,
    snowflake_table: str,
    data_file: str,
    ):
    sql_query = """
        COPY INTO {}.{} FROM @{}.%{}/{}.gz FILE_FORMAT = (TYPE=CSV, SKIP_HEADER=1, FIELD_OPTIONALLY_ENCLOSED_BY='"')
    """.format(
        snowflake_schema.upper(),
        snowflake_table.upper(),
        snowflake_schema.upper(),
        snowflake_table.upper(),
        data_file
        )
    return snowflake_client.execute_query(sql_query)



def upload_shopping_data(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_table: str,
    data_folder: str,
    data_file: str
    ):
    print("Uploading shopping data...")
    create_table(snowflake_client, snowflake_db, snowflake_schema, snowflake_table)
    use_database(snowflake_client, snowflake_db)
    stage_data(snowflake_client, snowflake_schema, snowflake_table, data_file, data_folder)
    copy_data(snowflake_client, snowflake_schema, snowflake_table, data_file)
    return


def prepare_shopping_data(
    folder: str,
    api_key: str,
    max_sessions: int
):
    """

    Each row in the dataset looks like:

    {'event_type': 'event_product',
    'hashed_url': '803f6c2d4202e39d6d7fdb232d69366b86bc843869c809f1e1954465bfc6e17f',
    'product_action': 'detail',
    'product_sku': '624bc145579b67b608e6a7b0d0516cc75e0ec4cbe44ec42c6ac53cc83925bc3e',
    'server_timestamp_epoch_ms': '1547528580651',
    'session_id': '0f1416c8c68bb9209c1bbc4576386df5480e9757f55ce9cb0d4d4017cf14fc1c'}

    """
    print("Preparing shopping data locally...")
    data_file = 'coveo_dataset_dump.csv'
    etl_timestamp = int(time.time() * 1000)
    etl_id = str(uuid.uuid4()) 
    coveo_dataset = CoveoDataset()
    # folder = '/Users/jacopotagliabue/Documents/repos/post-modern-data-stack-private/src'
    # pre-process only the training set as it's big enough already!
    with open(os.path.join(folder, data_file), 'w') as csvfile:
        # NOTE: this list has the same ordering as the CREATE TABLE statement
        fieldnames = ['etl_timestamp', 'etl_id', 'event_type', 'api_key', 'event_date', 'raw_data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        training_set = coveo_dataset.x_train[:max_sessions] if max_sessions else coveo_dataset.x_train
        print("Training set has {} sessions".format(len(training_set)))
        for session in training_set:
            for event in session:
                row = {}
                epoch_as_seconds = int(int(event['server_timestamp_epoch_ms']) / 1000.0)
                event_date = datetime.fromtimestamp(epoch_as_seconds).strftime("%Y-%m-%d") 
                row['etl_timestamp'] = etl_timestamp
                row['etl_id'] = etl_id
                row['event_date'] =  event_date
                row['api_key'] = api_key
                row['event_type'] = event['event_type']
                row['raw_data'] = json.dumps(event)
                writer.writerow(row)

    return data_file


def upload_data_to_snowflake(
    snowflake_client: SnowflakeClient,
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_table: str, 
    max_sessions: int, # IF SPECIFIED, ONLY USE THE FIRST N SESSIONS IN THE COVEO DATASET 
    api_key: str # simulate a typical b2b stack, this acts as a partition key on the append-only table
):
    print('Starting ops at {} '.format(datetime.utcnow()))
    # first, create schema
    create_schema(snowflake_client, snowflake_db, snowflake_schema)
    # create a temp dir to download dataset using recList and create 
    # a local file to bulk upload into Snowflake
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = prepare_shopping_data(tmpdirname, api_key, max_sessions)
        # create table if not there (drop it if there), upload data from CSV dump
        upload_shopping_data(
            snowflake_client,
            snowflake_db=snowflake_db,
            snowflake_schema=snowflake_schema,
            snowflake_table=snowflake_table,
            data_folder=tmpdirname,
            data_file=data_file
            )

    print('All done, see you, space cowboy {} '.format(datetime.utcnow()))
    return


if __name__ == "__main__":
    # init snowflake client
    sf_client = SnowflakeClient(
            user=os.getenv('SF_USER'),
            pwd=os.getenv('SF_PWD'),
            account=os.getenv('SF_ACCOUNT'),
            role=os.getenv('SF_ROLE'),
            keep_alive=False
            )
    # upload data from Coveo dataset
    upload_data_to_snowflake(
        snowflake_client=sf_client,
        snowflake_db=os.getenv('SF_DB'),
        snowflake_schema=os.getenv('SF_SCHEMA'),
        snowflake_table=os.getenv('SF_TABLE'),
        max_sessions=int(os.getenv('MAX_SESSIONS')) if os.getenv('MAX_SESSIONS') else None,
        api_key=os.getenv('APPLICATION_API_KEY')
    )