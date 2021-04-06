from __future__ import print_function
import os
import sys

from h2o.estimators.infogram import H2OInfogram

sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
    
def test_infogram_iris_plot():
    """
    Check to make sure infogram can be plotted
    :return: 
    """

    fr = h2o.import_file(path=pyunit_utils.locate("smalldata/admissibleml_test/irisROriginal.csv"))
    target = "Species"
    fr[target] = fr[target].asfactor()
    x = fr.names
    x.remove(target)
    
    infogram_model = H2OInfogram(seed = 12345, distribution = 'multinomial') # build infogram model with default settings
    infogram_model.train(x=x, y=target, training_frame=fr)
    infogram_model.plot(server=True) # make sure graph will not show

    infogram_model2 = H2OInfogram(seed = 12345, distribution = 'multinomial', cmi_threshold=0.05, relevance_threshold=0.05) # build infogram model with default settings
    infogram_model2.train(x=x, y=target, training_frame=fr)
    infogram_model2.plot(server=True)

    assert len(infogram_model.get_admissible_cmi()) <= len(infogram_model2.get_admissible_cmi())
    
if __name__ == "__main__":
    pyunit_utils.standalone_test(test_infogram_iris_plot)
else:
    test_infogram_iris_plot()
