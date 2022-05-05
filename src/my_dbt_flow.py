"""

We run a post-modern stack using:

    * Metaflow as the pipeline backbone;
    * Snowflake as data warehouse;
    * dbt as transformation layer;
    * AWS (Sagemaker) for PaaS deployment.


Please check the README and the companion blog post for the relevant background and context.

"""

from metaflow import FlowSpec, step, batch, S3, Parameter, current, Run, environment, card
from metaflow.cards import Table
from custom_decorators import enable_decorator, pip
import os
import subprocess
import json
import time
from datetime import datetime


try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")


class dbtFlow(FlowSpec):

    #NOTE: data parameters
    START_DATE = Parameter(
        name='start_date',
        help='Get data from this date, format yyyy-mm-dd',
        default='2019-01-13'
    )

    END_DATE = Parameter(
        name='end_date',
        help='Get data until this date, format yyyy-mm-dd',
        default='2019-03-14'
    )
    
    #NOTE: training parameters
    TRAINING_IMAGE = Parameter(
        name='training_image',
        help='AWS Docker Image URI for AWS Batch training',
        default='763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.8.0-gpu-py39-cu112-ubuntu20.04-e3'
    )

    #NOTE: endpoint deployment parameters
     # uri from: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    SERVING_IMAGE = Parameter(
        name='serving_image',
        help='AWS Docker Image URI for SageMaker Inference',
        default='763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.7.0-gpu-py38-cu112-ubuntu20.04-sagemaker'
    )

    # This is expensive! Remember to shut it down after completing the tutorial!
    # NOTE: by default, we delete the endpoint after testing for prediction
    # Comment out the delete command if you wish to use it!
    SAGEMAKER_INSTANCE = Parameter(
        name='sagemaker_instance',
        help='AWS Instance to Power SageMaker Inference',
        default='ml.p3.2xlarge'
    )

    # This is the name of the IAM role with SageMaker permissions
    # make sure this role has access to the bucket containing the tar file!
    IAM_SAGEMAKER_ROLE = Parameter(
        name='sagemaker_role',
        help='AWS Role for SageMaker',
        default='MetaSageMakerRole'
    )

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """
        # print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        if os.environ.get('EN_BATCH', '0') == '1':
            print("ATTENTION: AWS BATCH ENABLED!") 
        print(os.getenv('AWS_DEFAULT_REGION'))
        # make sure we can use the training image as an env
        os.environ['TRAIN_STEP_IMAGE'] = self.TRAINING_IMAGE
        # check variables and db connections are working fine
        assert os.environ['COMET_API_KEY']
        assert os.environ['SF_SCHEMA']
        assert os.environ['SF_TABLE']
        assert os.environ['SF_DB']
        # if dbt cloud is enabled, check for their values
        if bool(int(os.getenv('DBT_CLOUD'))):
            print("Flow will run with dbt cloud")
            assert int(os.getenv('DBT_ACCOUNT_ID'))
            assert int(os.getenv('DBT_PROJECT_ID'))
            assert int(os.getenv('DBT_JOB_ID'))
            assert os.getenv('DBT_API_KEY')
        from clients.snowflake_client import SnowflakeClient
        sf_client = SnowflakeClient(
            os.environ['SF_USER'],
            os.environ['SF_PWD'],
            os.environ['SF_ACCOUNT'],
            os.environ['SF_ROLE']
            )
        snowflake_version = sf_client.get_version()
        print(snowflake_version)
        assert snowflake_version is not None
        # check the data range makes sense
        start_date = datetime.strptime(self.START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(self.END_DATE, '%Y-%m-%d')
        assert end_date > start_date

        self.next(self.run_transformation)

    def get_dag_from_manifest(self, current_path):
        """
        This is a simple draft, for a function taking as input the manifest file
        produced by dbt and producing a dag-like structure compatible with the Metaflow
        API for card building.

        Starting from the code here to read the file: https://www.astronomer.io/blog/airflow-dbt-1/

        TODO: this is just a stub and it's not guaranteed (nor is intended) to generalize to
        arbitrary flows. It is included here as a bonus feature to showcase how deep the interplay
        between Metaflow and dbt can really be (thanks valay for the pointers!) 
        """
        local_filepath = current_path + "/dbt/target/manifest.json"
        # debug
        print(local_filepath)
        with open(local_filepath) as f:
            data = json.load(f)

        dbt_dag = {
            'start': {'type': 'start', 'box_next': True, 'box_ends': None, 'next': [], 'doc': 'Start!'},
            'end': {'type': 'end', 'box_next': True, 'box_ends': None, 'next': [], 'doc': 'See you, dbt cowboy'}
        }

        # fill the dag with data about the nodes
        for node in data["nodes"].keys():
            if node.split(".")[0] == "model":
                cnt_doc = data["nodes"][node]["description"]
                dbt_dag[node] = {'type': 'linear', 'box_next': True, 'box_ends': None, 'next': [], 'doc': cnt_doc} 

        # fill the dependencies
        for node in data["nodes"].keys():
            if node.split(".")[0] == "model":
                depends_on_nodes = data["nodes"][node]["depends_on"]["nodes"]
                # this is the first node, add id to start
                if not depends_on_nodes:
                    dbt_dag['start']['next'] = [node]
                for upstream_node in depends_on_nodes:
                    upstream_node_type = upstream_node.split(".")[0]
                    if upstream_node_type == "model":
                        dbt_dag[upstream_node]['next'] = [node]

        for node in dbt_dag.keys():
            # node not upstream of anything
            if node != 'end' and not dbt_dag[node]['next']:
                print("End node is: {}".format(node))
                dbt_dag[node]['next'] = ['end']
                break

        return dbt_dag

    @card(type='blank', id='dbtCard')
    @step
    def run_transformation(self):
        """
        Use dbt to transform raw data into tables / features
        """
        from clients.dbt_cloud_runner import dbtCloudRunner
        from pathlib import Path
        from metaflow.plugins.cards.card_modules.basic import DagComponent

        is_cloud = bool(int(os.getenv('DBT_CLOUD')))
        if is_cloud:
            # Run dbt in the cloud platform, using dbtCloudRunner as a 
            # wrapper around the cloud APIs
            print("Running with dbt cloud")
            # we basically trigger a job and then wait in a loop
            # until the status is SUCCESS or ERROR
            # NOTE: make sure the dbt credentials are in the env file
            # and that the job has been manually created in dbt cloud already!
            dbt_runner = dbtCloudRunner(
                account_id=int(os.getenv('DBT_ACCOUNT_ID')),
                project_id=int(os.getenv('DBT_PROJECT_ID')),
                job_id=int(os.getenv('DBT_JOB_ID')),
                cause='{}|{}|{}'.format(current.flow_name, current.run_id, current.username),
                dbt_cloud_api_key=os.getenv('DBT_API_KEY')
            )
            dbt_runner.run_job()
        else:
            my_dir = os.path.dirname(os.path.realpath(__file__)) 
            dbt_cmd = "dbt run --vars '{SF_SCHEMA: %s, SF_TABLE: %s}'"  % (os.getenv('SF_SCHEMA'), os.getenv('SF_TABLE'))
            # debug
            print("dir is {}, dbt command is: {}".format(my_dir, dbt_cmd))
            # run dbt from python
            process = subprocess.Popen([
                'source {}; {}'.format(os.path.join(os.getenv('VIRTUAL_ENV'), 'bin/activate'), dbt_cmd)
                ], cwd=os.path.join(my_dir, 'dbt/'), shell=True)
            process.communicate()
            if process.returncode != 0:
                raise Exception('dbt invocation returned exit code {}'.format(process.returncode))
            # build a card
            current_path = str(Path(__file__).resolve().parent)
            self.dbt_dag = self.get_dag_from_manifest(current_path)
            print(self.dbt_dag)
            current.card.append(DagComponent(data=self.dbt_dag))

        self.next(self.get_dataset)

    @step
    def get_dataset(self):
        """
        Get the data in the right shape from Snowflake, after the dbt transformation
        """
        import json
        from model.my_reclist import SessionDataset
        from clients.snowflake_client import SnowflakeClient
        sf_client = SnowflakeClient(
            os.environ['SF_USER'],
            os.environ['SF_PWD'],
            os.environ['SF_ACCOUNT'],
            os.environ['SF_ROLE']
        )
        # get data from Snowflake, returns lists of lists of SKUs, with no meta-data
        query = f"""
            SELECT 
                se."INTERACTIONS"
            FROM
                {os.getenv('SF_DB')}.{os.getenv('SF_SCHEMA')}.NEP_SESSION_EVENTS as se
            WHERE
                se."API_KEY"= %(api_key)s
                AND se."SESSION_DATE" > %(start_date)s
                AND se."SESSION_DATE" <= %(end_date)s
                AND ARRAY_SIZE(se."INTERACTIONS") > 2 -- we only retrieve sessions longer than 2 events
            ORDER BY se."SESSION_DATE" ASC -- sessions are ordered!
        """
        # debug
        print("Query to be exectued: {}".format(query))
        params = {
            'api_key': os.getenv('APPLICATION_API_KEY'),
            'start_date': self.START_DATE,
            'end_date': self.END_DATE
        }
        # fetch and snapshot raw dataset
        self.dataset = sf_client.fetch_all(query, params=params, debug=True)
        assert self.dataset
        # we split by time window, reserving only the last time span for testing
        # in this case, we simply take the last 10% of sessions
        # NOTE: data is ordered by time asc
        split_index = int(len(self.dataset) * 0.90)
        print("Split index is {}".format(split_index))
        # version also the train / test split and print some quick number
        # NOTE: "INTERACTIONS" are JSON-ified string, so we need to load them
        self.train_dataset = [json.loads(row['INTERACTIONS']) for row in self.dataset[:split_index]]
        self.test_dataset = [json.loads(row['INTERACTIONS']) for row in self.dataset[split_index:]]        
        print("# {} events in the training set, # {} in test set".format(
            len(self.train_dataset),
            len(self.test_dataset)
        ))
        # TODO: prepare train and test and initialize the dataset using reclist abstraction
        data = {}
        data["x_train"] = self.train_dataset
        data["x_test"] = [_[:-1] for _ in self.test_dataset]
        data["y_test"] = [[_[-1]] for _ in self.test_dataset]
        data["x_validation"] = [_[:-1] for _ in self.test_dataset]
        data["y_validation"] = [[_[-1]] for _ in self.test_dataset]
        data["catalog"] = {}
        self.session_dataset = SessionDataset(data=data)
        self.next(self.train_model)

    @environment(vars={
                    'EN_BATCH': os.getenv('EN_BATCH'),
                    'COMET_API_KEY': os.getenv('COMET_API_KEY')
                })
    # TODO: os.getenv may not work when we resume instead of starting from scracth
    @enable_decorator(batch(gpu=1, memory=80000, image=os.getenv('TRAIN_STEP_IMAGE')),
                      flag=os.getenv('EN_BATCH'))
    @pip(libraries={'reclist': '0.2.3', 'comet-ml': '3.26.0', 'numpy': '1.19.0'}) # numpy is there to avoid TF complaining
    @step
    def train_model(self):
        """
        Train models in parallel and store KPIs and path for downstream consumption

        """
        from comet_ml import Experiment
        from model.lstm_model import get_model
        import numpy as np
        from tensorflow.keras.optimizers import Adam # pylint: disable=import-error
        from tensorflow.keras.callbacks import EarlyStopping # pylint: disable=import-error 
        from tensorflow.keras.preprocessing.text import Tokenizer # pylint: disable=import-error
        from tensorflow.keras.preprocessing.sequence import pad_sequences # pylint: disable=import-error

        # TODO: pick a sensible EXP name!!!
        self.COMET_EXP_NAME = 'my_lstm_recs'
        # define some hyper parameters for model
        self.hypers = {
            'EMBEDDING_DIM': 32,
            'LSTM_HIDDEN_DIM': 32,
            'MAX_LEN': 20,
            'LEARNING_RATE': 1e-3,
            'DROPOUT': 0.3
        }
        # init comet object for tracking
        comet_exp = Experiment(project_name=self.COMET_EXP_NAME)
        # linking task to experiment
        comet_exp.add_tag(current.pathspec)
        comet_exp.log_parameters(self.hypers)

        # get sessions for training
        train_sessions = self.session_dataset.x_train
        # convert to strings for keras tokenization
        train_sessions = [' '.join(s) for s in train_sessions]
        # init tokenizer
        tokenizer = Tokenizer(
            filters='',
            lower=False,
            split=' ',
            oov_token='<UNK>'
        )
        # fit on training data to initialize vocab
        tokenizer.fit_on_texts(train_sessions)
        VOCAB_SIZE = len(tokenizer.word_index)
        # convert sessions to tokens
        train_sessions_token = tokenizer.texts_to_sequences(train_sessions)
        # get N-1 items as seed
        x_train = [s[:-1] for s in train_sessions_token]
        # pad to MAX_LEN
        x_train = np.array(pad_sequences(x_train, maxlen=self.hypers['MAX_LEN']))
        # get last item as label;
        # TODO: Decrementing index here because 0 is reserved for masking; Find a better way around this.
        y_train = np.array([ s[-1]-1 for s in train_sessions_token])
        print("NUMBER OF SESSIONS : {}".format(x_train.shape[0]))
        print('First 3 x:', x_train[:3])
        print('First 3 y:', y_train[:3])

        # get model
        model = get_model(VOCAB_SIZE,
                          self.hypers['MAX_LEN'],
                          self.hypers['EMBEDDING_DIM'],
                          self.hypers['LSTM_HIDDEN_DIM'],
                          self.hypers['DROPOUT'])
        # compile model
        model.compile(optimizer=Adam(learning_rate=self.hypers['LEARNING_RATE']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
        model.summary()
        # fit model
        model.fit(x_train, y_train,
                  epochs=100,
                  verbose=2,
                  batch_size=32,
                  validation_split=0.2,
                  callbacks=[EarlyStopping(patience=20)])
        comet_exp.end()
        # save model info
        self.model = {
            'model': model.to_json(),
            'model_weights': model.get_weights(),
            'tokenizer': tokenizer.to_json(),
            'model_config': {
                'vocab_size': VOCAB_SIZE,
                'max_len': self.hypers['MAX_LEN'],
                'embedding_dim': self.hypers['EMBEDDING_DIM'],
                'lstm_hidden_dim': self.hypers['LSTM_HIDDEN_DIM'],
                'dropout': self.hypers['DROPOUT']
            }
        }
        self.next(self.test_model)

    def build_table_from_rectests(self, test_results: list) -> Table:
        """
        Helper function looping over a list of test results from RecList
        and building a table to be displayed as a MF card.
        """
        header = ['Name', 'Description', 'Value']
        _table = [header]
        # TEST RESULTS have this shape
        # 'test_name': test.test_type
        # 'description': test.test_desc
        # 'test_result': test_result
        for result in test_results:
            # check if variable is float
            if not isinstance(result['test_result'], float):
                continue
            row = [
                result['test_name'],  
                result['description'],
                result['test_result']
            ]
            assert len(row) == len(header)
            _table.append(row)

        return Table(_table)

    @environment(vars={'EN_BATCH': os.getenv('EN_BATCH')})
    @enable_decorator(batch(gpu=1, memory=80000, image=os.getenv('TRAIN_STEP_IMAGE')),
                      flag=os.getenv('EN_BATCH'))
    @pip(libraries={'reclist': '0.2.3', 'numpy': '1.19.0'}) # numpy is there to avoid TF complaining
    @card(type='blank', id='recCard')
    @step
    def test_model(self):
        """
        Load the train model and use a custom RecList to test it
        and report its performance!
        """
        from model.lstm_model import LSTMRecModel
        from model.my_reclist import MyMetaflowRecList
        rec_model = LSTMRecModel(model_dict=self.model)
        y_preds = rec_model.predict(prediction_input=self.session_dataset.x_test)
        rec_list = MyMetaflowRecList(
          model=rec_model,
          dataset=self.session_dataset,
          y_preds=y_preds
        )
        rec_list(verbose=True)
        self.rec_results = rec_list.test_results
        # put rectests results into a table
        _table = self.build_table_from_rectests(self.rec_results)
        current.card['recCard'].append(_table)
        self.next(self.deploy_model)
        
    @step
    def deploy_model(self):
        """
        Use SageMaker to deploy the model as a stand-alone, PaaS endpoint, with our choice of the 
        underlying Docker image and hardware capabilities.

        Once the endpoint is deployed, you can add further infra, e.g. a lambda endpoint serving
        the predictions as a public services (https://github.com/jacopotagliabue/no-ops-machine-learning).
        Here, we just prove that the endpoint is up and running!
        """
        import shutil
        import tarfile
        from tensorflow.keras.models import model_from_json # pylint: disable=import-error

        # skip the deployment if not needed
        if not bool(int(os.getenv('SAGEMAKER_DEPLOY'))):
            print("Skipping deployment to Sagemaker")
        else:
            from sagemaker.tensorflow import TensorFlowModel
            # generate a signature for the endpoint, using timestamp as a convention
            ENDPOINT_NAME = 'nep-{}-endpoint'.format(int(round(time.time() * 1000)))
            print("\n\n================\nEndpoint name is: {}\n\n".format(ENDPOINT_NAME))

            # local temp file names
            model_name = "nep-model-{}/1".format(current.run_id)
            local_tar_name = 'model-{}.tar.gz'.format(current.run_id)
            # load model
            nep_model = model_from_json(self.model['model'])
            nep_model.set_weights(self.model['model_weights'])
            # save model locally
            nep_model.save(filepath=model_name)
            # save model as .tar.gz
            with tarfile.open(local_tar_name, mode="w:gz") as _tar:
                _tar.add(model_name, recursive=True)
            # remove local model
            shutil.rmtree(model_name.split('/')[0])
            # save model to S3 using metaflow S3 Client
            with open(local_tar_name, "rb") as in_file:
                data = in_file.read()
                with S3(run=self) as s3:
                    url = s3.put(local_tar_name, data)
                    print("Model saved at: {}".format(url))
                    # save this path for downstream reference!
                    self.model_s3_path = url
                    # remove local compressed model
                    os.remove(local_tar_name)
            # init sagemaker TF model
            model = TensorFlowModel(
               model_data=self.model_s3_path,
               image_uri=self.SERVING_IMAGE,
               role=self.IAM_SAGEMAKER_ROLE)
            # deploy sagemaker TF model
            predictor = model.deploy(
               initial_instance_count=1,
               instance_type=self.SAGEMAKER_INSTANCE,
               endpoint_name=ENDPOINT_NAME)
            # run a small test against the endpoint
            input = {'instances': [[10, 20, 30]]}
            # output is on the form {'predictions': [[0.0001, ..., 0.1283]]}
            result = predictor.predict(input)
            assert result['predictions'][0][0] > 0
            assert len(result['predictions'][0]) == self.model['model_config']['vocab_size']
            # print scores for first 10 prodcuts
            print(result['predictions'][0][:10])
            # delete the endpoint to avoid wasteful computing
            # NOTE: comment this if you want to keep it running
            # If deletion fails, make sure you delete the model in the console!
            print("Deleting endpoint now...")
            predictor.delete_endpoint()
            print("Endpoint deleted!")

        self.next(self.end)

    @step
    def end(self):
        """
        Just say bye!
        """
        print("All done\n\nSee you, space cowboy\n")
        return


if __name__ == '__main__':
    dbtFlow()
