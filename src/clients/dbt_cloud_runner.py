"""

Class handling the communication with dbt cloud through the REST API. This script is
a strip-down version of the Airflow operator from:

https://github.com/sungchun12/airflow-toolkit/blob/95d40ac76122de337e1b1cdc8eed35ba1c3051ed/dags/dbt_cloud_utils.py

NOTE: before using it, you need to first MANUALLY create a dbt Cloud job: 

https://docs.getdbt.com/docs/dbt-cloud/cloud-quickstart#create-a-new-job

As an example, a dbt Cloud job URL looks like:

https://cloud.getdbt.com/#/accounts/4238/projects/12220/jobs/12389/

"""

import time
import requests
from enum import Enum
from datetime import datetime


class dbtStatus(Enum):
    """
    Different dbt Cloud API status responses in integer format
    """
    QUEUED = 1
    STARTING = 2
    RUNNING = 3
    SUCCESS = 10
    ERROR = 20
    CANCELLED = 30


class dbtCloudRunner():
    """
    Utility to run dbt Cloud jobs.
    At init time, pass the relevant dbt cloud params
    """

    DTB_CLOUD_BASE_URL = 'https://cloud.getdbt.com'

    def __init__(
        self, 
        account_id: int,
        project_id: int,
        job_id: int,
        cause: str,
        dbt_cloud_api_key: str
    ) -> None:
        self.account_id = account_id
        self.project_id = project_id
        self.job_id = job_id
        self.cause = cause
        self.dbt_cloud_api_key = dbt_cloud_api_key

        return

    def _trigger_job(self) -> int:
        """
        Trigger the dbt Cloud job asynchronously, verifies variables
        match response payload from dbt Cloud api, return the job id,
        to be re-used for status tracking in the mail loop.
        """
        url = f"{self.DTB_CLOUD_BASE_URL}/api/v2/accounts/{self.account_id}/jobs/{self.job_id}/run/"
        headers = {"Authorization": f"Token {self.dbt_cloud_api_key}"}
        res = requests.post(
            url=url,
            headers=headers,
            data={
                "cause": f"{self.cause}", # properties of the Flow invoking the run
            },
        )

        try:
            res.raise_for_status()
        except:
            print(f"API token (last four): ...{self.dbt_cloud_api_key[-4:]}")
            raise

        response_payload = res.json()
        # debug dbt cloud response
        # print(response_payload["data"])
        # Verify the dbt Cloud job matches the arguments passed
        assert self.account_id == response_payload["data"]["account_id"]
        assert self.project_id == response_payload["data"]["project_id"]
        assert self.job_id == response_payload["data"]["job_definition_id"]

        return response_payload["data"]["id"]

    def _get_job_run_status(self, job_run_id: int) -> int:
        """
        Check job status based on job id
        """
        url = f"{self.DTB_CLOUD_BASE_URL}/api/v2/accounts/{self.account_id}/runs/{job_run_id}/"
        headers = {"Authorization": f"Token {self.dbt_cloud_api_key}"}
        res = requests.get(url=url, headers=headers)
        res.raise_for_status()
        response_payload = res.json()

        return response_payload["data"]["status"]

    # main function operator to trigger the job and a while loop to wait for success or error
    def run_job(self) -> None:
        """
        Main handler method to run the dbt Cloud job and track the job run status
        """
        job_run_id = self._trigger_job()
        # loop and check every 10 seconds the status of the job
        while True:
            job_run_status = self._get_job_run_status(job_run_id)
            print("Job status at {}: {}".format(datetime.utcnow(), job_run_status))
            if job_run_status == dbtStatus['SUCCESS'].value:
                print("dbt cloud job successfully completed at: {}".format(datetime.utcnow()))
                break
            elif job_run_status in (dbtStatus['ERROR'].value, dbtStatus['CANCELLED'].value):
                raise Exception("dbt cloud job failed at: {}".format(datetime.utcnow()))
            #sleep on it
            time.sleep(10)