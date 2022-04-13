from reclist.abstractions import RecList, rec_test, RecDataset


class MyMetaflowRecList(RecList):

    @rec_test(test_type='stats')
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self._y_preds,
                             self._y_test,
                             k=10)


class SessionDataset(RecDataset):
    """
    Wrapper around our session-based dataset
    """
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        super().__init__()

    def load(self):
        self._x_train = self.data["x_train"]
        self._y_train = None
        self._x_test = self.data["x_test"]
        self._y_test = self.data["y_test"]
        self._x_validation = self.data["x_validation"]
        self._y_validation = self.data["y_validation"]
        self._catalog = self.data["catalog"]

        return