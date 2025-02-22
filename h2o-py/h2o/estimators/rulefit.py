#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# This file is auto-generated by h2o-3/h2o-bindings/bin/gen_python.py
# Copyright 2016 H2O.ai;  Apache License Version 2.0 (see LICENSE for details)
#
from __future__ import absolute_import, division, print_function, unicode_literals

from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric


class H2ORuleFitEstimator(H2OEstimator):
    """
    RuleFit

    Builds a RuleFit on a parsed dataset, for regression or 
    classification. 
    """

    algo = "rulefit"
    supervised_learning = True

    def __init__(self,
                 model_id=None,  # type: Optional[Union[None, str, H2OEstimator]]
                 training_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
                 validation_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
                 seed=-1,  # type: int
                 response_column=None,  # type: Optional[str]
                 ignored_columns=None,  # type: Optional[List[str]]
                 algorithm="auto",  # type: Literal["auto", "drf", "gbm"]
                 min_rule_length=3,  # type: int
                 max_rule_length=3,  # type: int
                 max_num_rules=-1,  # type: int
                 model_type="rules_and_linear",  # type: Literal["rules_and_linear", "rules", "linear"]
                 weights_column=None,  # type: Optional[str]
                 distribution="auto",  # type: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace", "quantile", "huber"]
                 rule_generation_ntrees=50,  # type: int
                 auc_type="auto",  # type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]
                 ):
        """
        :param model_id: Destination id for this model; auto-generated if not specified.
               Defaults to ``None``.
        :type model_id: Union[None, str, H2OEstimator], optional
        :param training_frame: Id of the training data frame.
               Defaults to ``None``.
        :type training_frame: Union[None, str, H2OFrame], optional
        :param validation_frame: Id of the validation data frame.
               Defaults to ``None``.
        :type validation_frame: Union[None, str, H2OFrame], optional
        :param seed: Seed for pseudo random number generator (if applicable).
               Defaults to ``-1``.
        :type seed: int
        :param response_column: Response variable column.
               Defaults to ``None``.
        :type response_column: str, optional
        :param ignored_columns: Names of columns to ignore for training.
               Defaults to ``None``.
        :type ignored_columns: List[str], optional
        :param algorithm: The algorithm to use to generate rules.
               Defaults to ``"auto"``.
        :type algorithm: Literal["auto", "drf", "gbm"]
        :param min_rule_length: Minimum length of rules. Defaults to 3.
               Defaults to ``3``.
        :type min_rule_length: int
        :param max_rule_length: Maximum length of rules. Defaults to 3.
               Defaults to ``3``.
        :type max_rule_length: int
        :param max_num_rules: The maximum number of rules to return. defaults to -1 which means the number of rules is
               selected
               by diminishing returns in model deviance.
               Defaults to ``-1``.
        :type max_num_rules: int
        :param model_type: Specifies type of base learners in the ensemble.
               Defaults to ``"rules_and_linear"``.
        :type model_type: Literal["rules_and_linear", "rules", "linear"]
        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent
               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating
               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do
               not increase the size of the data frame. This is typically the number of times a row is repeated, but
               non-integer values are supported as well. During training, rows with higher weights matter more, due to
               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at
               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.
               Defaults to ``None``.
        :type weights_column: str, optional
        :param distribution: Distribution function
               Defaults to ``"auto"``.
        :type distribution: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",
               "quantile", "huber"]
        :param rule_generation_ntrees: specifies the number of trees to build in the tree model. Defaults to 50.
               Defaults to ``50``.
        :type rule_generation_ntrees: int
        :param auc_type: Set default multinomial AUC type.
               Defaults to ``"auto"``.
        :type auc_type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]
        """
        super(H2ORuleFitEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.seed = seed
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.algorithm = algorithm
        self.min_rule_length = min_rule_length
        self.max_rule_length = max_rule_length
        self.max_num_rules = max_num_rules
        self.model_type = model_type
        self.weights_column = weights_column
        self.distribution = distribution
        self.rule_generation_ntrees = rule_generation_ntrees
        self.auc_type = auc_type

    @property
    def training_frame(self):
        """
        Id of the training data frame.

        Type: ``Union[None, str, H2OFrame]``.
        """
        return self._parms.get("training_frame")

    @training_frame.setter
    def training_frame(self, training_frame):
        self._parms["training_frame"] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def validation_frame(self):
        """
        Id of the validation data frame.

        Type: ``Union[None, str, H2OFrame]``.
        """
        return self._parms.get("validation_frame")

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        self._parms["validation_frame"] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def seed(self):
        """
        Seed for pseudo random number generator (if applicable).

        Type: ``int``, defaults to ``-1``.
        """
        return self._parms.get("seed")

    @seed.setter
    def seed(self, seed):
        assert_is_type(seed, None, int)
        self._parms["seed"] = seed

    @property
    def response_column(self):
        """
        Response variable column.

        Type: ``str``.
        """
        return self._parms.get("response_column")

    @response_column.setter
    def response_column(self, response_column):
        assert_is_type(response_column, None, str)
        self._parms["response_column"] = response_column

    @property
    def ignored_columns(self):
        """
        Names of columns to ignore for training.

        Type: ``List[str]``.
        """
        return self._parms.get("ignored_columns")

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        assert_is_type(ignored_columns, None, [str])
        self._parms["ignored_columns"] = ignored_columns

    @property
    def algorithm(self):
        """
        The algorithm to use to generate rules.

        Type: ``Literal["auto", "drf", "gbm"]``, defaults to ``"auto"``.
        """
        return self._parms.get("algorithm")

    @algorithm.setter
    def algorithm(self, algorithm):
        assert_is_type(algorithm, None, Enum("auto", "drf", "gbm"))
        self._parms["algorithm"] = algorithm

    @property
    def min_rule_length(self):
        """
        Minimum length of rules. Defaults to 3.

        Type: ``int``, defaults to ``3``.
        """
        return self._parms.get("min_rule_length")

    @min_rule_length.setter
    def min_rule_length(self, min_rule_length):
        assert_is_type(min_rule_length, None, int)
        self._parms["min_rule_length"] = min_rule_length

    @property
    def max_rule_length(self):
        """
        Maximum length of rules. Defaults to 3.

        Type: ``int``, defaults to ``3``.
        """
        return self._parms.get("max_rule_length")

    @max_rule_length.setter
    def max_rule_length(self, max_rule_length):
        assert_is_type(max_rule_length, None, int)
        self._parms["max_rule_length"] = max_rule_length

    @property
    def max_num_rules(self):
        """
        The maximum number of rules to return. defaults to -1 which means the number of rules is selected
        by diminishing returns in model deviance.

        Type: ``int``, defaults to ``-1``.
        """
        return self._parms.get("max_num_rules")

    @max_num_rules.setter
    def max_num_rules(self, max_num_rules):
        assert_is_type(max_num_rules, None, int)
        self._parms["max_num_rules"] = max_num_rules

    @property
    def model_type(self):
        """
        Specifies type of base learners in the ensemble.

        Type: ``Literal["rules_and_linear", "rules", "linear"]``, defaults to ``"rules_and_linear"``.
        """
        return self._parms.get("model_type")

    @model_type.setter
    def model_type(self, model_type):
        assert_is_type(model_type, None, Enum("rules_and_linear", "rules", "linear"))
        self._parms["model_type"] = model_type

    @property
    def weights_column(self):
        """
        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the
        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative
        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data
        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.
        During training, rows with higher weights matter more, due to the larger loss function pre-factor. If you set
        weight = 0 for a row, the returned prediction frame at that row is zero and this is incorrect. To get an
        accurate prediction, remove all rows with weight == 0.

        Type: ``str``.
        """
        return self._parms.get("weights_column")

    @weights_column.setter
    def weights_column(self, weights_column):
        assert_is_type(weights_column, None, str)
        self._parms["weights_column"] = weights_column

    @property
    def distribution(self):
        """
        Distribution function

        Type: ``Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",
        "quantile", "huber"]``, defaults to ``"auto"``.
        """
        return self._parms.get("distribution")

    @distribution.setter
    def distribution(self, distribution):
        assert_is_type(distribution, None, Enum("auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace", "quantile", "huber"))
        self._parms["distribution"] = distribution

    @property
    def rule_generation_ntrees(self):
        """
        specifies the number of trees to build in the tree model. Defaults to 50.

        Type: ``int``, defaults to ``50``.
        """
        return self._parms.get("rule_generation_ntrees")

    @rule_generation_ntrees.setter
    def rule_generation_ntrees(self, rule_generation_ntrees):
        assert_is_type(rule_generation_ntrees, None, int)
        self._parms["rule_generation_ntrees"] = rule_generation_ntrees

    @property
    def auc_type(self):
        """
        Set default multinomial AUC type.

        Type: ``Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]``, defaults to
        ``"auto"``.
        """
        return self._parms.get("auc_type")

    @auc_type.setter
    def auc_type(self, auc_type):
        assert_is_type(auc_type, None, Enum("auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"))
        self._parms["auc_type"] = auc_type


    def rule_importance(self):
        """
        Retrieve rule importances for a Rulefit model

        :return: H2OTwoDimTable
        """
        if self._model_json["algo"] != "rulefit":
            raise H2OValueError("This function is available for Rulefit models only")
        return self._model_json["output"]['rule_importance']
