package hex.Infogram;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelBuilderHelper;
import hex.ModelCategory;
import water.H2O;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.TwoDimTable;
import water.DKV;

import java.util.*;

import hex.genmodel.utils.DistributionFamily;

import static hex.Infogram.InfogramUtils.*;
import static hex.gam.MatrixFrameUtils.GamUtils.keepFrameKeys;



public class Infogram extends ModelBuilder<hex.Infogram.InfogramModel, hex.Infogram.InfogramModel.InfogramParameters,
        hex.Infogram.InfogramModel.InfogramModelOutput> {
  boolean _buildCore;       // true to find core predictors, false to find admissible predictors
  String[] _topKPredictors; // contain the names of top predictors to consider for infogram
  Frame _baseOrSensitiveFrame = null;
  String[] _modelDescription; // describe each model in terms of predictors used
  int _numModels;             // number of models to build
  double[] _cmi;              // store conditional mutual information
  double[] _cmiRaw;           // conditional mutual information before normalization to 0 and 1
  TwoDimTable _varImp;
  int _numPredictors;         // number of predictors to consider in training dataset
  Key<Frame> _cmiRelKey;      // store frame key containing column, admissibe, admissible_index, relevance, cmi, cmi_raw
  List<Key<Frame>> _generatedFrameKeys; // keep track of all keys generated

  public Infogram(boolean startup_once) { super(new hex.Infogram.InfogramModel.InfogramParameters(), startup_once);}

  public Infogram(hex.Infogram.InfogramModel.InfogramParameters parms) {
    super(parms);
    init(false);
  }

  public Infogram(hex.Infogram.InfogramModel.InfogramParameters parms, Key<hex.Infogram.InfogramModel> key) {
    super(parms, key);
    init(false);
  }

  @Override
  protected Driver trainModelImpl() {
    return new InfogramDriver();
  }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[] { ModelCategory.Binomial, ModelCategory.Multinomial};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive)
      validateInfoGramParameters();
  }

  private void validateInfoGramParameters() {
    Frame dataset = _parms.train();

    if (!_parms.train().vec(_parms._response_column).isCategorical()) // only classification is supported now
      error("response_column", " only classification is allowed.  Change your response column " +
              "to be categorical before calling Infogram.");

    // make sure protected attributes are true predictor columns
    if (_parms._protected_columns != null) {
      Set<String> colNames = new HashSet<>(Arrays.asList(dataset.names()));
      for (String senAttribute : _parms._protected_columns)
        if (!colNames.contains(senAttribute))
          error("protected_columns", "protected_columns: "+senAttribute+" is not a valid " +
                  "column in the training dataset.");
    }

    _buildCore = _parms._protected_columns == null;
    
    if (_buildCore) {
      if (_parms._net_information_threshold == -1) { // not set
        _parms._cmi_threshold = 0.1;
        _parms._net_information_threshold = 0.1;
      } else if (_parms._net_information_threshold > 1 || _parms._net_information_threshold < 0) {
        error("net_information_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._cmi_threshold = _parms._net_information_threshold;
      }

      if (_parms._total_information_threshold == -1) {  // not set
        _parms._relevance_threshold = 0.1;
        _parms._total_information_threshold = 0.1;
      } else if (_parms._total_information_threshold < 0 || _parms._total_information_threshold > 1) {
        error("total_information_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._relevance_threshold = _parms._total_information_threshold;
      }
      
      if (_parms._safety_index_threshold != -1) {
        warn("safety_index_threshold", "Should not set safety_index_threshold for core infogram " +
                "runs.  Set net_information_threshold instead.  Using default of 0.1 if not set");
      }
      
      if (_parms._relevance_index_threshold != -1) {
        warn("relevance_index_threshold", "Should not set relevance_index_threshold for core " +
                "infogram runs.  Set total_information_threshold instead.  Using default of 0.1 if not set");
      }
    } else { // fair infogram
      if (_parms._safety_index_threshold == -1) {
        _parms._cmi_threshold = 0.1;
        _parms._safety_index_threshold = 0.1;
      } else if (_parms._safety_index_threshold < 0 || _parms._safety_index_threshold > 1) {
        error("safety_index_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._cmi_threshold = _parms._safety_index_threshold;
      }
      
      if (_parms._relevance_index_threshold == -1) {
        _parms._relevance_threshold = 0.1;
        _parms._relevance_index_threshold = 0.1;
      } else if (_parms._relevance_index_threshold < 0 || _parms._relevance_index_threshold > 1) {
        error("relevance_index_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._relevance_threshold = _parms._relevance_index_threshold;
      }
      
      if (_parms._net_information_threshold != -1) {
        warn("net_information_threshold", "Should not set net_information_threshold for fair " +
                "infogram runs, set safety_index_threshold instead.  Using default of 0.1 if not set");
      }
      if (_parms._total_information_threshold != -1) {
        warn("total_information_threshold", "Should not set total_information_threshold for fair" +
                " infogram runs, set relevance_index_threshold instead.  Using default of 0.1 if not set");
      }
    }

    // check top k to be between 0 and training dataset column number
    if (_parms._top_n_features < 0)
      error("_topk", "topk must be between 0 and the number of predictor columns in your training dataset.");

    _numPredictors = _parms.train().numCols()-1;
    if (_parms._weights_column != null)
      _numPredictors--;
    if (_parms._offset_column != null)
      _numPredictors--;
    if ( _parms._top_n_features > _numPredictors) {
      warn("_topk", "topk exceed the actual number of predictor columns in your training dataset." +
              "  It will be set to the number of predictors in your training dataset.");
      _parms._top_n_features = _numPredictors;
    }

    if (_parms._nparallelism < 0)
      error("nparallelism", "must be >= 0.  If 0, it is adaptive");

    if (_parms._nparallelism == 0) // adaptively set nparallelism
      _parms._nparallelism = H2O.NUMCPUS;
    
    if (_parms._compute_p_values)
      error("compute_p_values", " compute_p_values calculation is not yet implemented.");
    
    if (nclasses() < 2)
      error("distribution", " infogram currently only supports classification models");
    
    if (DistributionFamily.AUTO.equals(_parms._distribution)) {
      _parms._distribution = (nclasses() == 2) ? DistributionFamily.bernoulli : DistributionFamily.multinomial;
    }
  }

  private class InfogramDriver extends Driver {
    void generateBasicFrame() {
      String[] eligiblePredictors = extractPredictors(_parms);  // exclude senstive attributes if applicable
      _baseOrSensitiveFrame = extractTrainingFrame(_parms, _parms._protected_columns, 1, _parms.train().clone());
      _parms.fillImpl(); // copy over model specific parameters to build infogram
      _topKPredictors = extractTopKPredictors(_parms, _parms.train(), eligiblePredictors, _generatedFrameKeys); // extract topK predictors
    }

    @Override
    public void computeImpl() {
      init(true);
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(Infogram.this);
      _job.update(0, "Initializing model training");
      _generatedFrameKeys = new ArrayList<>(); // generated infogram model plus one for safe Infogram
      generateBasicFrame(); // generate training frame with predictors and sensitive features (if specified)
      _numModels = 1 + _topKPredictors.length;
      _modelDescription = generateModelDescription(_topKPredictors, _parms._protected_columns);
      buildModel();
    }
    // todo:  add max_runtime_secs restrictions
    public final void buildModel() {
      hex.Infogram.InfogramModel model = null;
      try {
        model = new hex.Infogram.InfogramModel(dest(), _parms, 
                new hex.Infogram.InfogramModel.InfogramModelOutput(Infogram.this));
        model.delete_and_lock(_job);
        model._output._start_time = System.currentTimeMillis();
        _cmiRaw = new double[_numModels];
        buildInfoGramsNRelevance(); // calculate relevance, cmi
        _job.update(1, "finished building models for Infogram ...");
        model._output.setDistribution(_parms._distribution);
        model._output.copyCMIRelevance(_cmiRaw, _cmi, _topKPredictors, _varImp, _parms._cmi_threshold,
                _parms._relevance_threshold); // copy over cmi, relevance of all predictors
        _cmi = model._output._cmi;
        _cmiRelKey = model._output.generateCMIRelFrame(_buildCore);
        model._output.extractAdmissibleFeatures(_varImp, model._output._all_predictor_names, _cmi, _cmiRaw,
                _parms._cmi_threshold, _parms._relevance_threshold);  // extract admissible information model output
        _job.update(1, "Infogram building completed...");
        model.update(_job._key);
      } finally {
        DKV.remove(_baseOrSensitiveFrame._key);
        removeFromDKV(_generatedFrameKeys);
        final List<Key> keep = new ArrayList<>();
        if (model != null) {
          keepFrameKeys(keep, _cmiRelKey);
        }
        Scope.exit(keep.toArray(new Key[keep.size()]));
        model.update(_job._key);
        model.unlock(_job);
      }
    }

    /***
     * Top level method to break down the infogram process into parts.
     * 
     * I have a question here for all of you:  Instead of generating the training frame and model builders for all
     * the predictors, I break this down into several parts with each part generating _parms._nparallelism training
     * frames and model builders.  For each part, after _parms._nparallelism models are built, I extract the entropy
     * for each predictor.  Then, I move to the next part.  My question here is:  is this necessary?  I am afraid of
     * the memory consumption of spinning up so many training frames and model builders.  If this is not an issue,
     * please let me know.
     */
    private void buildInfoGramsNRelevance() {
      int outerLoop = (int) Math.floor(_numModels/_parms._nparallelism); // last model is build special
      int modelCount = 0;
      int lastModelInd = _numModels - 1;
      if (outerLoop > 0) {  // build parallel models but limit it to parms._nparallelism at a time
        for (int outerInd = 0; outerInd < outerLoop; outerInd++) {
          buildModelCMINRelevance(modelCount, _parms._nparallelism, lastModelInd);
          modelCount += _parms._nparallelism;
          _job.update(_parms._nparallelism, "in the middle of building infogram models.");
        }
      }
      int leftOver = _numModels - modelCount;
      if (leftOver > 0) { // finish building the leftover models
        buildModelCMINRelevance(modelCount, leftOver, lastModelInd);
        _job.update(leftOver, " building the final set of infogram models.");
      }
      _cmi = calculateFinalCMI(_cmiRaw, _buildCore);  // calculate cmi from entropy and scale cmi to be from 0 to 1
    }

    /***
     * This method basically go through all the predictors and calculate the cmi associated with each predictor.  For
     * core infogram, refer to https://h2oai.atlassian.net/browse/PUBDEV-8075 section I.  For fair infogram, refer to
     * https://h2oai.atlassian.net/browse/PUBDEV-8075 section II.
     * 
     * @param modelCount
     * @param numModel
     * @param lastModelInd
     */
    private void buildModelCMINRelevance(int modelCount, int numModel, int lastModelInd) {
      boolean lastModelIndcluded = (modelCount+numModel >= lastModelInd);
      Frame[] trainingFrames = buildTrainingFrames(_topKPredictors, _parms.train(), _baseOrSensitiveFrame, modelCount,
              numModel, _buildCore, lastModelInd, _generatedFrameKeys); // generate training frame
      Model.Parameters[] modelParams = buildModelParameters(trainingFrames, _parms._infogram_algorithm_parameters,
              numModel, _parms._infogram_algorithm); // generate parameters
      ModelBuilder[] builders = ModelBuilderHelper.trainModelsParallel(buildModelBuilders(modelParams),
              numModel); // build models in parallel
      if (lastModelIndcluded) // extract relevance here for core infogram
        extractRelevance(builders[numModel-1].get(), modelParams[numModel-1]);
      generateInfoGrams(builders, trainingFrames, _cmiRaw, modelCount, numModel, _parms._response_column, 
              _generatedFrameKeys); // extract model, score, generate infogram
    }

    /**
     * For core infogram, the last model is the one with all predictors.  In this case, the relevance is basically the
     * variable importance.  For fair infogram, the last model is the one with all the predictors minus the protected
     * columns.  Again, the relevance is the variable importance.
     * 
     * @param model
     * @param parms
     */
    private void extractRelevance(Model model, Model.Parameters parms) {
      if (_buildCore) { // full model is last one, just extract varImp
        _varImp = extractVarImp(_parms._infogram_algorithm, model); // extract relevance as variable importance
      } else {  // need to build model for fair info grame
        Frame fullFrame = subtractAdd2Frame(_baseOrSensitiveFrame, _parms.train(), _parms._protected_columns,
                _topKPredictors); // training frame is topKPredictors minus protected_columns
        parms._train = fullFrame._key;
        _generatedFrameKeys.add(fullFrame._key);
        ModelBuilder builder = ModelBuilder.make(parms);
        Model fairModel = (Model) builder.trainModel().get();
        _varImp = extractVarImp(_parms._infogram_algorithm, fairModel);
        Scope.track_generic(fairModel);
      }
    }
  }
}
