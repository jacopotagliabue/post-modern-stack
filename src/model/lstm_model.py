import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reclist.abstractions import RecModel


class LSTMRecModel(RecModel):

    def __init__(self, model_dict, **kwargs):
        super().__init__(**kwargs)
        self.model_dict = model_dict
        self.tokenizer = tokenizer_from_json(self.model_dict['tokenizer'])
        self._model = model_from_json(self.model_dict['model'])
        self._model.set_weights(self.model_dict['model_weights'])

    def predict(self, prediction_input: list, *args, **kwargs):
        """
        The predict function should implement the behaviour of the model at inference time.

        :param prediction_input: the input that is used to to do the prediction
        :param args:
        :param kwargs:
        :return:
        """
        k = kwargs.get('k', 10)
        predictions = []
        sessions_as_text = [' '.join(_) for _ in prediction_input]
        sessions_tokenized = self.tokenizer.texts_to_sequences(sessions_as_text)
        sessions_padded = np.array(pad_sequences(sessions_tokenized,
                                                 maxlen=self.model_dict['model_config']['max_len']))
        bs = 128
        for idx in range(0, len(sessions_padded), bs):
            batch_sessions = sessions_padded[idx: idx+bs]
            raw_predictions = self.model.predict(batch_sessions, batch_size=128)
            sorted_indices = np.argsort(raw_predictions, axis=-1)[:, -k:][:, ::-1]
            for idx in range(len(sorted_indices)):
                predictions.append([self.tokenizer.index_word[i+1] for i in sorted_indices[idx]])

        return predictions


def get_model(vocab_size, max_len, embedding_dim, lstm_hidden_dim, dropout):
    # define LSTM model
    model = Sequential()
    # use vocab_size + 1 if mask_zero is True
    model.add(Embedding(vocab_size + 1,
                        embedding_dim,
                        input_length=max_len,
                        mask_zero=True))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_hidden_dim))
    model.add(Dense(vocab_size, activation='softmax'))
    return model
