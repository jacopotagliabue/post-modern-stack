# The Post-Modern Stack

Joining the modern data stack with the modern ML stack

## Overview

As part our TDS series on MLOps, our [blog post](https://towardsdatascience.com/the-post-modern-stack-993ec3b044c1) shows how a _post-modern stack_ works, by deconstructing (see the pun?) our original [YDNABB](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) repo into the few fundamental pieces owning the actual compute: a data warehouse for dataOps, and Metaflow on AWS for MLOps. A quick, high-level walk-through of the stack can be found in our intro video:

[![YouTube intro video](/images/youtube.png)](https://www.youtube.com/watch?v=5kHDb-XGHtc)

As a use case, we pick a popular RecSys challenge, session-based recommendation: given the interactions between a shopper and some products in a browsing session, can we train a model to predic what the next interaction will be? The flow is powered by our open-source [Coveo Data Challenge dataset](https://github.com/coveooss/SIGIR-ecom-data-challenge) - as model, we train a vanilla LSTM, a model just complex enough to make good use of cloud computing. At a quick glance, this is what we are building:

![The post-modern stack](/images/stack.jpg)

As usual, we show a working, end-to-end, real-world flow: while you can run it locally with few thousands sessions to get the basics, we suggest you to use the `MAX_SESSIONS` variable to appreciate how well the stack scales - with no code changes - as millions of events are pushed to the warehouse.

For an in-depth explanation of the philosophy behind the approach, please check the companion [blog post](https://towardsdatascience.com/the-post-modern-stack-993ec3b044c1), and the previous episodes / repos in [the series](https://towardsdatascience.com/tagged/mlops-without-much-ops). 

## Pre-requisites

The code is a self-contained recommender project; however, since we leverage best-in-class tools, some preliminary setup is required. Please make sure the requirements are satisfied, depending on what you wish to run - roughly in order of ascending complexity:

_The basics: Metaflow, Snowflake and dbt_

A Snowflake account is needed to host the data, and a working Metaflow + dbt setup is needed to run the flow; we *strongly* suggest to run `Metaflow on AWS` (as it is the intended setup), but with some minor modifications you should be able to run the flow with a local store as well. 

* _Snowflake account_: [sign-up for a free trial](https://signup.snowflake.com).
* _AWS account_: [sign-up for a free AWS account](https://aws.amazon.com/free/).
* _Metaflow on AWS_: [follow the setup guide](https://docs.metaflow.org/metaflow-on-aws).
* _dbt core setup_: on top of installing the open source package (already included in the `requirements.txt`), you need to point dbt to your Snowflake instance with the proper [dbt_profile](https://docs.getdbt.com/dbt-cli/configure-your-profile), Make sure the SCHEMA there matches with what is specified in the `.env` file (`SF_SCHEMA`).

_Adding experiment tracking_

* _Comet ML_: [sign-up for free](https://www.comet.ml/signup) and get an api key. If you don't want experiment tracking, make sure to comment out the Comet specific parts in the `train_model` step.

_Adding PaaS deployment_

* _SageMaker setup_: To deploy the model as a PaaS solution using SageMaker, the `IAM_SAGEMAKER_ROLE` parameter in the flow needs to contain a suitable IAM ROLE to deploy an endpoint and access the s3 bucket where Metaflow is storing the model artifact; if you don't wish to deploy your model, run the flow with `SAGEMAKER_DEPLOY=0` in the `.env` file.

_Adding dbt cloud_

* _dbt cloud account_: [sign-up for free](https://www.getdbt.com/signup) and get an api key. If you don't wish to use dbt cloud but just the local setup,set `DBT_CLOUD=0` in the `.env` file.

## Setup

### Virtual env

Setup a virtual environment with the project dependencies:

* `python -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`

NOTE: the current version of RecList has some old dependencies which may results in some (harmless) pip conflicts - conflicts will disappear with the new version, coming out soon.

Create a local version of the `local.env` file named only `.env` (do _not_ commit it!), and make sure to fill its values properly:

| VARIABLE | TYPE | MEANING |
| ------------- | ------------- | ------------- |
| SF_USER | string  | Snowflake user name  |
| SF_PWD | string |  Snowflake password  |
| SF_ACCOUNT | string  |  Snowflake account  |
| SF_DB | string |  Snowflake database  |
| SF_SCHEMA | string (suggested: POST_MODERN_DATA_STACK) |  Snowflake schema for raw and transformed data  |
| SF_TABLE | string (COVEO_DATASET_RAW)  |  Snowflake table for raw data  |
| SF_ROLE | string |  Snowflake role to run SQL |
| APPLICATION_API_KEY | uuid (474d1224-e231-42ed-9fc9-058c2a8347a5)  | Organization id to simulate a SaaS company  |
| MAX_SESSIONS | int (1000)  | Number of raw sessions to load into Snowflake (try first running the project locally with a small number) |
| EN_BATCH | 0-1 (0)  | Enable/disable cloud computing for @batch steps in Metaflow (try first running the project locally) |
| COMET_API_KEY | string  | Comet ML api key  |
| DBT_CLOUD| 0-1 (0)  | Enable/disable running dbt on the cloud |
| SAGEMAKER_DEPLOY| 0-1 (1)  | Enable/disable deploying the model artifact to a Sagemaker endpoint  |
| DBT_ACCOUNT_ID | int  | dbt cloud account id (you can find it in the dbt cloud URL)  |
| DBT_PROJECT_ID | int  | dbt cloud project id  (you can find it in the dbt cloud URL) |
| DBT_JOB_ID | int  | dbt cloud job id (you can find it in the dbt cloud URL) |
| DBT_API_KEY| string  | dbt cloud api key  |

### Load data into Snowflake 

Original datasets are from the Coveo SIGIR Data Challenge. To save you from downloading the original data dump and dealing with large text files, we re-used the abstraction over the data provided by RecList. If you run `upload_to_snowflake.py` in the `upload` folder from your laptop as a one-off script, the program will download the Data Challenge dataset and dump it to a Snowflake table that simulates the [append-only log pattern](https://towardsdatascience.com/the-modern-data-pattern-d34d42216c81). This allows us to use dbt and Metaflow to run a realistic ELT and ML code over real-world data.

Once you run the script, check your Snowflake for the new schema/table:

![Raw table in Snowflake](/images/raw_table.png)

If you wish to see how a data ingestion pipeline works (i.e. an endpoint streaming into Snowlake real-time, individual events, instead of a bulk upload), we open-sourced a [serverless pipeline](https://github.com/jacopotagliabue/paas-data-ingestion) as well.

### dbt

While we will run dbt code as part of Metaflow, it is good practice to try and see if everything works from a stand-alone setup first. To run and test the dbt transformations, just `cd` into the `dbt` folder and run `dbt run --vars '{SF_SCHEMA: POST_MODERN_DATA_STACK, SF_TABLE: COVEO_DATASET_RAW}'`, where the [variables](https://docs.getdbt.com/docs/building-a-dbt-project/building-models/using-variables) reflect the content of your `.env` file  (you can also run `dbt test`, if you like).

Once you run dbt, check your Snowflake for the views:

![Views in Snowflake](/images/after_dbt.png)

The `DBT_CLOUD` variable (see above) controls whether transformations are run from _within the flow folder_, or from a dbt cloud account, by using dbt API to trigger the transformation on the cloud platform. If you want to leverage dbt cloud, make sure to manually [create a job](https://docs.getdbt.com/docs/dbt-cloud/cloud-quickstart#create-a-new-job) on the platform, and then configure the relevant variables in the `.env` file. In our tests, we used the exact same `.sql` and `.yml` files that you find in this repository:

<img src="/images/dbt_cloud.png" height="250">

Please note that instead of having a local dbt folder, you could have your dbt code in a Github repo and then either clone it using Github APIs at runtime, or import it in dbt cloud and use the platform to run the code base.

## How to run (a.k.a. the whole enchilada)

### Run the flow

Once the above setup steps are completed, you can run the flow:

* cd into the `src` folder;
* run the flow with `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_dbt_flow.py --package-suffixes ".py" run --max-workers 4`, where `METAFLOW_PROFILE` is needed to select a specific Metaflow config (you can omit it, if you're using the default), `AWS_PROFILE` is needed to select a specific AWS config that runs the flow and it's related AWS infrastructure (you can omit it, if you're using the default), and `AWS_DEFAULT_REGION` is needed to specify the target AWS region (you can omit it, if you've it already specified in your local AWS PROFILE and you do not wish to change it);
* visualize the preformance card with `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_dbt_flow.py card view test_model --id recCard` (see below for an intro to [RecList](https://github.com/jacopotagliabue/reclist)).

### Results

If you run the fully-featured flow (i.e. `SAGEMAKER_DEPLOY=1`) with the recommended setup, you will end up with:

* an up-to-date view in Snowflake, leveraging dbt to make raw data ready for machine learning; 
* versioned datasets and model artifacts in your AWS, accessible through the standard [Metaflow client API](https://docs.metaflow.org/metaflow/client);
* a Comet dashboard for experiment tracking of the deep learning model, displaying training stats;
* a versioned Metaflow card containing (some of) the tests run with RecList (see below);
*  finally, a DL-based, sequential recommender system serving predictions in real-time using SageMaker for inference.

If you log in into your AWS SageMaker interface, you should find the new endpoint for next event prediction available for inference:

 ![aws sagemaker UI](/images/aws_sagemaker.png)

If you run the flow with dbt cloud, you will also find the dbt run in the history section on the cloud platform, easily identifiable through the flow id and user. 

 ![dbt run history](/images/dbt_run_history.png)


### BONUS: RecList and Metaflow cards

The project includes a (stub of a) custom [DAG card](https://outerbounds.com/blog/integrating-pythonic-visual-reports-into-ml-pipelines/) showing how the model is performing according to [RecList](https://github.com/jacopotagliabue/reclist), our open-source framework for behavioral testing. We could devote an [article](https://towardsdatascience.com/ndcg-is-not-all-you-need-24eb6d2f1227) / [paper](https://arxiv.org/abs/2111.09963) just to this (as we actually did recently!); you can visualize it with `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_dbt_flow.py card view test_model --id recCard` at the end of your run. No matter how small, we wanted to include the card/test as a reminder of _how important is to understand model behavior before deployment_. Cards are a natural UI to display some of the RecList information: since readable, shareable (self-)documentation is crucial for production, RecList new major release will include out-of-the-box support for visualization and reporting tools: reach out if you're interested!

As a *bonus* bonus feature (thanks Valay for the snippet!), *only when running with the dbt core setup*, the (not-production-ready) function `get_dag_from_manifest` will read the local manifest file and produce a dictionary compatible with Metaflow Card API. If you type `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_dbt_flow.py card view run_transformation --id dbtCard` at the end of a successful run, you should see a card displaying the dbt card _as a Metaflow card_, as in the image below:

 ![dbt card on Metaflow](/images/dbt_card.png)

 We leave to the reader (and / or to future iterations) to explore how to combine dbt, RecList and other info into a custom, well-designed card!

## What's next?

Of course, the post-modern stack can be further expanded or improved in many ways. Without presumption of completeness, these are some ideas to start:

* on the dataOps side, we could include some data quality checks, either by improving our dbt setup, or by introducing additional tooling: at [reasonable scale](https://towardsdatascience.com/hagakure-for-mlops-the-four-pillars-of-ml-at-reasonable-scale-5a09bd073da) the greater marginal value is typically to be found in better data, as compared to better models;
* on the MLOps side, we barely scracthed the surface: one side, we kept the modeling simple and avoid any tuning, which is however very easy to do using Metaflow buit-in parallelization abilities; on the other, you may decide to complicate the flow with other tools, improve on serving etc. (e.g. the proposal [here](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat)). Swapping in-and-out different tools with similar functionalities should be easy: in a previous work, we [abstracted away experiment tracking](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat/blob/main/local_flow/rec/src/utils.py) and allow users to pick [Neptune](https://neptune.ai/) as an alternative SaaS platform. Similar considerations apply to this use case as well;
* a proper RecList for this flow is yet to be developed, as the current proposal is nothing more than a stub showing how easy it is to run a devoted test suite when needed: you can augment the simple suite we prepared, improve the visualization on cards or both - since RecList roadmap is quickly progressing, we expect a deeper integration and a whole new set of functionalities to be announced soon. Stay tuned for our next iteration on this!

Is this the *only* way to run dbt in Metaflow? Of course not - in particular, you could think of writing a small wrapper around a flow and a dbt-core project that creates individual Metaflow steps corresponding to individual dbt steps, pretty much like suggested [here](https://www.astronomer.io/blog/airflow-dbt-1/) for another orchestrator. But this is surely a story for another repo / time ;-)

## Acknowledgements

Special thanks to Sung Won Chung from dbt Labs, Hugo Bowne-Anderson, Gaurav Bhushan, Savin Goyal, Valay Dave from Outerbounds, Luca Bigon, Andrea Polonioli and Ciro Greco from Coveo. 

If you liked this project and the related article, please take a second to add a star to this and our [RecList](https://github.com/jacopotagliabue/reclist) repository!

Contributors:

* [Jacopo Tagliabue](https://www.linkedin.com/in/jacopotagliabue/), general design, Metaflow fan boy, prototype.
* [Patrick John Chia](https://www.linkedin.com/in/patrick-john-chia/), model, deployment and testing.

## License

All the code in this repo is freely available under a MIT License, also included in the project.
