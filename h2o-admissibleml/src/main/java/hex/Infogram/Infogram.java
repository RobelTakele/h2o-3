package hex.Infogram;

import hex.*;
import water.H2O;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.TwoDimTable;
import water.DKV;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import hex.genmodel.utils.DistributionFamily;

import static hex.Infogram.InfogramModel.InfogramModelOutput.sortCMIRel;
import static hex.Infogram.InfogramUtils.*;
import static hex.gam.MatrixFrameUtils.GamUtils.keepFrameKeys;
import static water.util.ArrayUtils.sort;
import static water.util.ArrayUtils.sum;


public class Infogram extends ModelBuilder<hex.Infogram.InfogramModel, hex.Infogram.InfogramModel.InfogramParameters,
        hex.Infogram.InfogramModel.InfogramModelOutput> {
  boolean _buildCore;       // true to find core predictors, false to find admissible predictors
  String[] _topKPredictors; // contain the names of top predictors to consider for infogram
  Frame _baseOrSensitiveFrame = null;
  String[] _modelDescription; // describe each model in terms of predictors used
  int _numModels; // number of models to build
  double[] _cmi;  // store conditional mutual information
  double[] _cmiValid;
  double[] _cmiCV;
  double[] _cmiRaw;  // raw conditional mutual information
  double[] _cmiRawValid;  // raw conditional mutual information from validation frame
  double[] _cmiRawCV;
  String[] _columnsCV;
  TwoDimTable _varImp;
  int _numPredictors; // number of predictors in training dataset
  Key<Frame> _cmiRelKey;
  Key<Frame> _cmiRelKeyValid;
  Key<Frame> _cmiRelKeyCV;
  List<Key<Frame>> _generatedFrameKeys; // keep track of all keys generated
  boolean _cvDone = false;  // on when we are inside cv
  private transient InfogramModel _model;
  long _validNonZeroNumRows;
  int _nFoldOrig = 0;
  Model.Parameters.FoldAssignmentScheme _foldAssignmentOrig = null;
  String _foldColumnOrig = null;
  
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
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds,2);
  }

  /***
   * This is called before cross-validation is carried out
   */
  @Override
  public void computeCrossValidation() {
    info("cross-validation", "cross-validation infogram information is stored in frame with key" +
            " labeled as relevance_cmi_key_cv and the admissible features in admissible_features_cv.");
    if (error_count() > 0) {
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(Infogram.this);
    }
    super.computeCrossValidation();
  }

  // find the best alpha/lambda values used to build the main model moving forward by looking at the devianceValid
  @Override
  public void cv_computeAndSetOptimalParameters(ModelBuilder[] cvModelBuilders) {
    int nBuilders = cvModelBuilders.length;
    double[][] cmiRaw = new double[nBuilders][];
    ArrayList<ArrayList<String>> columns = new ArrayList<ArrayList<String>>();
    long[] nObs = new long[nBuilders];
    for (int i = 0; i < cvModelBuilders.length; ++i) {  // run cv for each lambda value
      InfogramModel g = (InfogramModel) cvModelBuilders[i].dest().get();
      Scope.track_generic(g);
      extractInfogramInfo(g, cmiRaw, columns, i);
      nObs[i] = g._output._validNonZeroNumRows;
    }
    calculateMeanInfogramInfo(cmiRaw, columns, nObs);
    for (int i = 0; i < cvModelBuilders.length; ++i) {
      Infogram g = (Infogram) cvModelBuilders[i];
      InfogramModel gm = g._model;
      gm.write_lock(_job);
      gm.update(_job);
      gm.unlock(_job);
    }
    _cvDone = true;  // cv is done and we are going to build main model next
  }
  
  
  public void calculateMeanInfogramInfo(double[][] cmiRaw, ArrayList<ArrayList<String>> columns,
                                        long[] nObs) {
    int nFolds = cmiRaw.length;
    ArrayList<String> oneFoldColumns = columns.get(0);  // get column names of first fold
    _columnsCV = oneFoldColumns.toArray(new String[oneFoldColumns.size()]);
    int nPreds = cmiRaw[0].length;
    _cmiCV = new double[nPreds];
    _cmiRawCV = new double[nPreds];
    double oneOverNObsSum = 1.0/sum(nObs);
    for (int fIndex = 0; fIndex < nFolds; fIndex++) {  // get sum of each fold
      ArrayList<String> oneFoldC = columns.get(fIndex);
      for (int pIndex = 0; pIndex < nPreds; pIndex++) { // go through each predictor
        String colName = oneFoldColumns.get(pIndex);    // use same predictor order as zero fold
        int currFoldIndex = oneFoldC.indexOf(colName);  // current fold colName order index change
        _cmiRawCV[pIndex] += cmiRaw[fIndex][currFoldIndex] * nObs[fIndex] * oneOverNObsSum;
      }
    }
    // normalize CMI and relevane again
    double maxCMI = ArrayUtils.maxValue(_cmiRawCV);
    double oneOverMaxCMI = maxCMI == 0 ? 0 : 1.0/maxCMI;
    for (int pIndex = 0; pIndex < nPreds; pIndex++) {
      _cmiCV[pIndex] = _cmiRawCV[pIndex]*oneOverMaxCMI;
    }
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
      List<String> colNames = Arrays.asList(dataset.names());
      for (String senAttribute : _parms._protected_columns)
        if (!colNames.contains(senAttribute))
          error("protected_columns", "protected_columns: "+senAttribute+" is not a valid " +
                  "column in the training dataset.");
    }

    _buildCore = _parms._protected_columns == null;
    
    if (_buildCore) {
      if (_parms._net_information_threshold == Double.MIN_VALUE) {
        _parms._cmi_threshold = 0.1;
        _parms._net_information_threshold = 0.1;
      } else if (_parms._net_information_threshold > 1 || _parms._net_information_threshold < 0) {
        error("net_information_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._cmi_threshold = _parms._net_information_threshold;
      }

      if (_parms._total_information_threshold == Double.MIN_VALUE) {
        _parms._relevance_threshold = 0.1;
        _parms._total_information_threshold = 0.1;
      } else if (_parms._total_information_threshold < 0 || _parms._total_information_threshold > 1) {
        error("total_information_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._relevance_threshold = _parms._total_information_threshold;
      }
      
      if (_parms._safety_index_threshold != Double.MIN_VALUE) {
        warn("safety_index_threshold", "Should not set safety_index_threshold for core infogram " +
                "runs.  Set net_information_threshold instead.  Using default of 0.1 if not set");
      }
      
      if (_parms._relevance_index_threshold != Double.MIN_VALUE) {
        warn("relevance_index_threshold", "Should not set relevance_index_threshold for core " +
                "infogram runs.  Set total_information_threshold instead.  Using default of 0.1 if not set");
      }
    } else { // fair infogram
      if (_parms._safety_index_threshold == Double.MIN_VALUE) {
        _parms._cmi_threshold = 0.1;
        _parms._safety_index_threshold = 0.1;
      } else if (_parms._safety_index_threshold < 0 || _parms._safety_index_threshold > 1) {
        error("safety_index_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._cmi_threshold = _parms._safety_index_threshold;
      }
      
      if (_parms._relevance_index_threshold == Double.MIN_VALUE) {
        _parms._relevance_threshold = 0.1;
        _parms._relevance_index_threshold = 0.1;
      } else if (_parms._relevance_index_threshold < 0 || _parms._relevance_index_threshold > 1) {
        error("relevance_index_threshold", " should be set to be between 0 and 1.");
      } else {
        _parms._relevance_threshold = _parms._relevance_index_threshold;
      }
      
      if (_parms._net_information_threshold != Double.MIN_VALUE) {
        warn("net_information_threshold", "Should not set net_information_threshold for fair " +
                "infogram runs, set safety_index_threshold instead.  Using default of 0.1 if not set");
      }
      if (_parms._total_information_threshold != Double.MIN_VALUE) {
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
    
    if (_cvDone) {  // disable cv now that we are in main model
      _nFoldOrig = _parms._nfolds;
      _foldColumnOrig = _parms._fold_column;
      _foldAssignmentOrig = _parms._fold_assignment;
      _parms._fold_column = null;
      _parms._nfolds = 0;
      _parms._fold_assignment = null;
    }
  }

  private class InfogramDriver extends Driver {
    void generateBasicFrame() {
      String[] eligiblePredictors = extractPredictors(_parms, _foldColumnOrig);  // exclude senstive attributes if applicable
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
      generateBasicFrame();            // generate tranining frame with predictors and sensitive features (if specified)
      _numModels = 1 + _topKPredictors.length;
      _modelDescription = generateModelDescription(_topKPredictors, _parms._protected_columns);
      buildModel();
    }
    // todo:  add max_runtime_secs restrictions
    public final void buildModel() {
      try {
        boolean validPresent = _parms.valid() != null;
        _model = new hex.Infogram.InfogramModel(dest(), _parms, new hex.Infogram.InfogramModel.InfogramModelOutput(Infogram.this));
        _model.delete_and_lock(_job);
        _model._output._start_time = System.currentTimeMillis();
        _cmiRaw = new double[_numModels];
        if (_parms.valid() != null)
          _cmiRawValid = new double[_numModels];
        buildInfoGramsNRelevance(validPresent); // calculate mean CMI
        _job.update(1, "finished building models for Infogram ...");
        _model._output.setDistribution(_parms._distribution);
        _model._output.copyCMIRelevance(_cmiRaw, _cmi, _topKPredictors, _varImp, _parms._cmi_threshold,
                _parms._relevance_threshold); // copy over cmi, relevance of all predictors to model._output
        _cmi = _model._output._cmi;
        
        if (validPresent)
          _model._output.copyCMIRelevanceValid(_cmiRawValid, _cmiValid, _varImp, _parms._cmi_threshold,
                  _parms._relevance_threshold); // copy over cmi, relevance of all predictors to model._output
        
        _cmiRelKey = _model._output.setCMIRelFrame(validPresent, _buildCore);
        _model._output.extractAdmissibleFeatures(DKV.getGet(_cmiRelKey), false, false);
        if (validPresent) {
          _cmiRelKeyValid = _model._output._relCmiKey_valid;
          _model._output.extractAdmissibleFeatures(DKV.getGet(_cmiRelKeyValid), true, false);
          _model._output._validNonZeroNumRows = _validNonZeroNumRows;
        }
        if (_cvDone) {                       // CV is enabled and now we are in main model
          _cmiRelKeyCV = setCMIRelFrameCV(_buildCore); // generate relevance and CMI frame from cv runs
          _model._output._relCmiKey_xval = _cmiRelKeyCV;
          _model._output._relevance_cmi_key_xval = _cmiRelKeyCV.toString();
          _model._output.extractAdmissibleFeatures(DKV.getGet(_cmiRelKeyCV), false, true);
          _parms._nfolds = _nFoldOrig;
          _parms._fold_assignment = _foldAssignmentOrig;
          _parms._fold_column = _foldColumnOrig;
        }
        
        _job.update(1, "Infogram building completed...");
        _model.update(_job._key);
      } finally {
        DKV.remove(_baseOrSensitiveFrame._key);
        removeFromDKV(_generatedFrameKeys);
        final List<Key> keep = new ArrayList<>();

        if (_model != null) {
          keepFrameKeys(keep, _cmiRelKey);
          if (_cmiRelKeyValid != null)
            keepFrameKeys(keep, _cmiRelKeyValid);
          if (_cmiRelKeyCV != null)
            keepFrameKeys(keep, _cmiRelKeyCV);
        }
        Scope.exit(keep.toArray(new Key[keep.size()]));
        _model.update(_job._key);
        _model.unlock(_job);
      }
    }
    
    private Key<Frame> setCMIRelFrameCV(boolean buildCore) {
      int nPred = _columnsCV.length;
      double[] admissibleIndex = new double[nPred];
      double[] admissible = new double[nPred];
      ArrayList<String> mainModelPredNames = new ArrayList<>(Arrays.asList(_model._output._all_predictor_names));
      double[] mainModelRelevance = _model._output._relevance;
      double[] relevanceCV = new double[mainModelRelevance.length];
     
      for (int index=0; index<nPred; index++) {
        String colName = _columnsCV[index];
        int mainIndex = mainModelPredNames.indexOf(colName);
        relevanceCV[index] = mainModelRelevance[mainIndex];
        double temp1 = 1-relevanceCV[index];
        double temp2 = 1-_cmiCV[index];
        admissibleIndex[index] = Math.sqrt(temp1*temp1+temp2*temp2);
        admissible[index] = _cmiCV[index]>=_parms._cmi_threshold && relevanceCV[index]>=_parms._relevance_threshold 
                ? 1 : 0;
      }
      int[] indices = IntStream.range(0, _cmiCV.length).toArray();
      sort(indices, admissibleIndex, -1, 1);
      sortCMIRel(indices, relevanceCV, _cmiRawCV, _cmiCV, _columnsCV, admissibleIndex, admissible);
      Frame cmiRelFrame = generateCMIRelevance(_columnsCV, admissible, admissibleIndex, relevanceCV, _cmiCV,
              _cmiRawCV, buildCore);
      return cmiRelFrame._key;
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
     * 
     * @param validPresent true if there is a validation dataset
     */
    private void buildInfoGramsNRelevance(boolean validPresent) {
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
      _cmi = calculateFinalCMI(_cmiRaw, _buildCore);  // scale cmi to be from 0 to 1, ignore last one
      if (validPresent)
        _cmiValid = calculateFinalCMI(_cmiRawValid, _buildCore);
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
              numModel);      // build models in parallel
      if (lastModelIndcluded) // extract relevance here for core infogram
        extractRelevance(builders[numModel-1].get(), modelParams[numModel-1]);
      _validNonZeroNumRows = generateInfoGrams(builders, trainingFrames, _parms.valid(), _cmiRaw, _cmiRawValid, modelCount,
              numModel, _parms._response_column, _generatedFrameKeys); // extract model, score, generate infogram
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
      if (_buildCore) {           // full model is last one, just extract varImp
        _varImp = extractVarImp(_parms._infogram_algorithm, model);
      } else {                    // need to build model for fair info grame
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
