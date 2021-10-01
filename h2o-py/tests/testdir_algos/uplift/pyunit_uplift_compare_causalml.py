import sys
import os
sys.path.insert(1, os.path.join("..", "..", ".."))
import h2o
from tests import pyunit_utils
from h2o.estimators.uplift_random_forest import H2OUpliftRandomForestEstimator
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.dataset import make_uplift_classification
from causalml.metrics import auuc_score
import numpy as np


def uplift_compare():
    df, feature_cols = make_uplift_classification(n_samples=5000, 
                                                  treatment_name=["control", "treatment"],
                                                  n_classification_features=10,
                                                  n_classification_informative=10,
                                                  random_seed=42)
    
    treatment_column = "treatment_group_key"
    response_column = "conversion"

    mask = np.random.rand(len(df)) < 0.5
    train_df, test_df = df[mask].copy(deep=True), df[~mask].copy(deep=True)

    train = h2o.H2OFrame(train_df)
    train[treatment_column] = train[treatment_column].asfactor()
    train[response_column] = train[response_column].asfactor()

    test = h2o.H2OFrame(test_df)
    test[treatment_column] = test[treatment_column].asfactor()
    test[response_column] = test[response_column].asfactor()

    train_df[treatment_column].replace({1: "treatment", 0: "control"}, inplace=True)
    
    ntree = 10
    max_depth = 5
    
    auuc_types = ["qini", "lift", "gain"]
    h2o_drfs = [None] * len(auuc_types)
    for i in range(len(auuc_types)):
        drf = H2OUpliftRandomForestEstimator(
            ntrees=ntree,
            max_depth=max_depth,
            treatment_column=treatment_column,
            uplift_metric="KL",
            distribution="bernoulli",
            gainslift_bins=10,
            min_rows=5,
            nbins=1000,
            seed=42,
            auuc_type=auuc_types[i],
            sample_rate=0.9
        )
        drf.train(y=response_column, x=feature_cols, training_frame=train)
        h2o_drfs[i] = drf
        
    df[treatment_column] = df[treatment_column].astype(str)
    uplift_model = UpliftRandomForestClassifier(
        n_estimators=ntree,
        max_depth=max_depth,
        evaluationFunction="KL",
        control_name="control",
        min_samples_leaf=5,
        min_samples_treatment=5,
        normalization=False,
        random_state=42,
    )
    uplift_model.fit(
        train_df[feature_cols].values,
        treatment=train_df[treatment_column].values,
        y=train_df[response_column].values
    )
    
    testing_df = test
    causal_preds = uplift_model.predict(test_df.values)

    for i in range(len(h2o_drfs)):
        preds_h2o = h2o_drfs[i].predict(testing_df)
    
        preds_comp = preds_h2o["uplift_predict"]
        preds_comp.names = ["h2o"]
        preds_comp["causal"] = h2o.H2OFrame(causal_preds)
        preds_comp["diff"] = abs(preds_comp["h2o"] - preds_comp["causal"])
        preds_comp[treatment_column] = testing_df[treatment_column]
        preds_comp[response_column] = testing_df[response_column]
        preds_comp.summary()
    
        mean_diff = preds_comp["diff"].mean(return_frame=False)[0]
        
        print("Average difference: %f" % mean_diff)
        assert mean_diff < 0.1, str(mean_diff)+": Average difference should not be higher than 10 %"

    results = preds_comp.as_data_frame()
    results = results[["h2o", "causal", response_column, treatment_column]]
    mapping = {'control': 0, 'treatment': 1}
    results = results.replace({treatment_column: mapping})

    # calculate auuc using CausalML package
    auuc = auuc_score(results, outcome_col=response_column, treatment_col=treatment_column, normalize=False)
        
    # compare AUUC calculation with CausalML
    h2o_auuc_qain_test = h2o_drfs[2].model_performance(testing_df).auuc()
    print("AUUC calculation:")
    diff = abs(auuc["h2o"] - h2o_auuc_qain_test)
    print("CausalML H2O: %f H2O: %f diff: %f" % (auuc["h2o"], h2o_auuc_qain_test, diff))
    assert diff < 7, \
        "Absolute difference between causalML package and H2O AUUC calculation is higher than is expected: %f" % diff
    
    diff = abs(auuc["causal"] - h2o_auuc_qain_test)
    print("CausalML: %f H2O: %f diff: %f" % (auuc["causal"], h2o_auuc_qain_test, diff))
    assert diff < 7,\
        "Absolute difference between causalML package and H2O AUUC is higher than is expected: %f" % diff
    
    # test plot_auuc
    perf = h2o_drfs[0].model_performance(testing_df)
    n, uplift = perf.plot_auuc(metric="gain", plot=False)
    print(uplift)
    
    
if __name__ == "__main__":
    pyunit_utils.standalone_test(uplift_compare)
else:
    uplift_compare()
