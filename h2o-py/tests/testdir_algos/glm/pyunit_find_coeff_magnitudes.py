from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import numpy as np

# check varimp for Binomial, Multinomial, Regression when standardize is set to False.
# I did the following:
# 1. train GLM with a dataset with standardize = True
# 2. train GLM with a dataset with standardized numerical columns and with standardize = False
#
# The standardized coefficients from model 1 and the coefficients from model 2 should be the same in this case.
def test_standardized_coeffs():
    # binomial
    h2o_data = h2o.import_file(pyunit_utils.locate("smalldata/glm_test/binomial_20_cols_10KRows.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C3"] = h2o_data["C3"].asfactor()
    h2o_data["C4"] = h2o_data["C4"].asfactor()
    h2o_data["C5"] = h2o_data["C5"].asfactor()
    h2o_data["C6"] = h2o_data["C6"].asfactor()
    h2o_data["C7"] = h2o_data["C7"].asfactor()
    h2o_data["C8"] = h2o_data["C8"].asfactor()
    h2o_data["C9"] = h2o_data["C9"].asfactor()
    h2o_data["C10"] = h2o_data["C10"].asfactor()
    y = "C21"
    x = h2o_data.names
    x.remove(y)
    h2o_data["C21"] = h2o_data["C21"].asfactor()
    h2o_model = H2OGeneralizedLinearEstimator(family="binomial")
    h2o_model.train(x=x, y=y, training_frame=h2o_data)
    print(h2o_model.coef())
    np.save('/Users/wendycwong/temp/binomial_coeffs', h2o_model.coef())
    h2o_data = h2o.import_file(pyunit_utils.locate("smalldata/glm_test/gaussian_20cols_10000Rows.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C3"] = h2o_data["C3"].asfactor()
    h2o_data["C4"] = h2o_data["C4"].asfactor()
    h2o_data["C5"] = h2o_data["C5"].asfactor()
    h2o_data["C6"] = h2o_data["C6"].asfactor()
    h2o_data["C7"] = h2o_data["C7"].asfactor()
    h2o_data["C8"] = h2o_data["C8"].asfactor()
    h2o_data["C9"] = h2o_data["C9"].asfactor()
    h2o_data["C10"] = h2o_data["C10"].asfactor()
    y = "C21"
    x = h2o_data.names
    x.remove(y)
    h2o_model = H2OGeneralizedLinearEstimator(family="gaussian")
    h2o_model.train(x=x, y=y, training_frame=h2o_data)
    print(h2o_model.coef())
    np.save('/Users/wendycwong/temp/gaussian_coeffs', h2o_model.coef())
    h2o_data = h2o.import_file(pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C3"] = h2o_data["C3"].asfactor()
    h2o_data["C4"] = h2o_data["C4"].asfactor()
    h2o_data["C5"] = h2o_data["C5"].asfactor()
    y = "C11"
    x = h2o_data.names
    x.remove(y)
    h2o_data["C11"] = h2o_data["C11"].asfactor()
    h2o_model = H2OGeneralizedLinearEstimator(family="multinomial")
    h2o_model.train(x=x, y=y, training_frame=h2o_data)
    print(h2o_model.coef())
    np.save('/Users/wendycwong/temp/multinomial_coeffs', h2o_model.coef())

if __name__ == "__main__":
  pyunit_utils.standalone_test(test_standardized_coeffs)
else:
    test_standardized_coeffs()
