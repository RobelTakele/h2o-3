package hex.Infogram;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import hex.*;
import hex.deeplearning.DeepLearningModel;
import hex.genmodel.utils.DistributionFamily;
import hex.glm.GLMModel;
import hex.schemas.*;
import hex.tree.drf.DRFModel;
import hex.tree.gbm.GBMModel;
import hex.tree.xgboost.XGBoostModel;
import water.*;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;
import water.udf.CFuncRef;
import water.util.TwoDimTable;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.IntStream;

import static hex.Infogram.InfogramModel.InfogramParameters.Algorithm.glm;
import static hex.Infogram.InfogramUtils.generateCMIRelevance;
import static hex.genmodel.utils.DistributionFamily.*;
import static hex.glm.GLMModel.GLMParameters.Family.binomial;
import static hex.util.DistributionUtils.distributionToFamily;
import static water.util.ArrayUtils.sort;

public class InfogramModel extends Model<InfogramModel, InfogramModel.InfogramParameters, InfogramModel.InfogramModelOutput> {
  /**
   * Full constructor
   *
   * @param selfKey
   * @param parms
   * @param output
   */
  public InfogramModel(Key<InfogramModel> selfKey, InfogramParameters parms, InfogramModelOutput output) {
    super(selfKey, parms, output);
  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    assert domain == null;
    switch(_output.getModelCategory()) {
      case Binomial:
        return new ModelMetricsBinomial.MetricBuilderBinomial(domain);
      case Multinomial:
        return new ModelMetricsMultinomial.MetricBuilderMultinomial(_output.nclasses(), domain, _parms._auc_type);
      default:
        throw H2O.unimpl("Invalid ModelCateory "+_output.getModelCategory());
    }
  }

  @Override
  protected PredictScoreResult predictScoreImpl(Frame fr, Frame adaptFrm, String destination_key, Job j,
                                                boolean computeMetrics, CFuncRef customMetricFunc) {
    throw new UnsupportedOperationException("Infogram does not support scoring on data.  It only provides information" +
            " on predictors and choose admissible features for users.  Users can take the admissible features, build" +
            "their own model and score with that model.");
  }
  
  @Override
  protected double[] score0(double[] data, double[] preds) {
    throw new UnsupportedOperationException("Infogram does not support scoring on data.  It only provides information" +
            " on predictors and choose admissible features for users.  Users can take the admissible features, build" +
            "their own model and score with that model.");
  }

  @Override
  public Frame score(Frame fr, String destinationKey, Job j, boolean computeMetrics, CFuncRef customMetricFunc) {
    throw new UnsupportedOperationException("Infogram does not support scoring on data.  It only provides information" +
            " on predictors and choose admissible features for users.  Users can take the admissible features, build" +
            "their own model and score with that model.");
  }

  public static class InfogramParameters extends Model.Parameters {
    public Algorithm _infogram_algorithm = Algorithm.gbm;     // default to GBM
    public String _infogram_algorithm_params = new String();  // store user specific parameters for chosen algorithm
    public String[] _protected_columns = null;    // store features to be excluded from final model
    public double _cmi_threshold = 0.1;           // default set by Deep
    public double _relevance_threshold = 0.1;         // default set by Deep
    public double _total_information_threshold = Double.MIN_VALUE;  // relevance threshold for core infogram
    public double _net_information_threshold = Double.MIN_VALUE;    // cmi threshold for core infogram
    public double _safety_index_threshold = Double.MIN_VALUE;       // cmi threshold for safe infogram
    public double _relevance_index_threshold = Double.MIN_VALUE;    // relevance threshold for safe infogram
    public double _data_fraction = 1.0;               // fraction of data to use to calculate infogram
    public Model.Parameters _infogram_algorithm_parameters;   // store parameters of chosen algorithm
    public int _top_n_features = 50;                          // if 0 consider all predictors, otherwise, consider topk predictors
    public boolean _compute_p_values = false;                 // if true, will calculate p-value
    public int _nparallelism = 0;
    
    public enum Algorithm {
      AUTO,
      deeplearning,
      drf,
      gbm,
      glm,
      xgboost
    }

    @Override
    public String algoName() {
      return "Infogram";
    }

    @Override
    public String fullName() {
      return "Information Diagram";
    }

    @Override
    public String javaName() {
      return InfogramModel.class.getName();
    }

    @Override
    public long progressUnits() {
      return 0;
    }

    /**
     * This method performs the following functions:
     * 1. extract the algorithm specific parameters from _algorithm_params to _algorithm_parameters which will be 
     * one of GBMParameters, DRFParameters, DeepLearningParameters, GLMParameters.
     * 2. Next, it will copy the parameters that are common to all algorithms from InfogramParameters to 
     * _algorithm_parameters.
     */
    /**
     * This method performs the following functions:
     * 1. it will extract the algorithm specific parameters from _info_algorithm_params to
     * infogram_algorithm_parameters which will be one of GBMParameters, DRFParameters, DeepLearningParameters or 
     * GLMParameters.  This will be used to build models and extract the infogram.
     * 2. Next, it will copy the parameters that are common to all algorithms from InfogramParameters to 
     * _algorithm_parameters.
     */
    public void fillImpl() {
      Properties p = new Properties();
      boolean fillParams;
      List<String> excludeList = new ArrayList<>(); // prevent overriding of parameters set by user
      fillParams = _infogram_algorithm_params != null && !_infogram_algorithm_params.isEmpty();

      if (fillParams) { // only execute when algorithm specific parameters are filled in by user
        HashMap<String, String[]> map =
                new Gson().fromJson(_infogram_algorithm_params, new TypeToken<HashMap<String, String[]>>() {
                }.getType());
        for (Map.Entry<String, String[]> param : map.entrySet()) {
          String[] paramVal = param.getValue();
          String paramName = param.getKey();
          excludeList.add("_" + paramName);
          if (paramVal.length == 1) {
            p.setProperty(paramName, paramVal[0]);
          } else {
            p.setProperty(paramName, Arrays.toString(paramVal));
          }
        }
      }

      ModelParametersSchemaV3 paramsSchema;
      Model.Parameters params;
      Algorithm algoName = _infogram_algorithm;
      switch (algoName) {
        case glm:
          paramsSchema = new GLMV3.GLMParametersV3();
          params = new GLMModel.GLMParameters();
          excludeList.add("_distribution");
          ((GLMModel.GLMParameters) params)._family = distributionToFamily(this._distribution);
          break;
        case AUTO: // auto defaults to GBM
        case gbm:
          paramsSchema = new GBMV3.GBMParametersV3();
          params = new GBMModel.GBMParameters();
          if (!excludeList.contains("_stopping_tolerance")) {
            params._stopping_tolerance = 0.01;  // set default to 0.01
            excludeList.add("_stopping_tolerance");
          }
          break;
        case drf:
          paramsSchema = new DRFV3.DRFParametersV3();
          params = new DRFModel.DRFParameters();
          if (!excludeList.contains("_stopping_tolerance")) {
            params._stopping_tolerance = 0.01;  // set default to 0.01
            excludeList.add("_stopping_tolerance");
          }
          break;
        case deeplearning:
          paramsSchema = new DeepLearningV3.DeepLearningParametersV3();
          params = new DeepLearningModel.DeepLearningParameters();
          break;
        case xgboost:
          paramsSchema = new XGBoostV3.XGBoostParametersV3();
          params = new XGBoostModel.XGBoostParameters();
          break;
        default:
          throw new UnsupportedOperationException("Unknown algo: " + algoName);
      }

      paramsSchema.init_meta();
      _infogram_algorithm_parameters = (Model.Parameters) paramsSchema
              .fillFromImpl(params)
              .fillFromParms(p, true)
              .createAndFillImpl();

      copyInfoGramParams(excludeList); // copy over InfogramParameters that are applicable to model specific algos
    }

    public void copyInfoGramParams(List<String> excludeList) {
      Field[] algoParams = Model.Parameters.class.getDeclaredFields();
      Field algoField;
      for (Field oneField : algoParams) {
        try {
          String fieldName = oneField.getName();
          algoField = this.getClass().getField(fieldName);
          if (excludeList.size() == 0 || !excludeList.contains(fieldName)) {
            algoField.set(_infogram_algorithm_parameters, oneField.get(this));
          }
        } catch (IllegalAccessException | NoSuchFieldException e) { // suppress error printing.  Only care about fields that are accessible
          ;
        }
      }
    }
  }

  public static class InfogramModelOutput extends Model.Output {
    final public static int _COLUMN_INDEX=0;
    final public static int _ADMISSIBLE_PREDICTOR_INDEX =1;
    final public static int _RELEVANCE_INDEX=3;
    final public static int _CMI_INDEX=4;
    final public static int _CMI_RAW_INDEX=5;
    public double[] _admissible_cmi;  // conditional info for admissible features in _admissible_features
    public double[] _admissible_cmi_raw;  // conditional info for admissible features in _admissible_features raw
    public String[] _admissible_features; // predictors chosen that exceeds both conditional_info and varimp thresholds
    public String[] _admissible_features_valid;
    public String[] _admissible_features_xval;
    public double[] _admissible_index;  // store distance from 1,1 corner of infogram plot
    public double[] _admissible_index_valid; // needed to build validation frame
    public double[] _admissible; // 0 if predictor is admissible and 1 otherwise
    public double[] _admissible_valid;
    public double[] _cmi_raw; // cmi before normalization and for all predictors
    public double[] _cmi_raw_valid;
    public double[] _cmi; // normalized cmi
    public double[] _cmi_valid;
    public double[] _admissible_relevance;  // varimp values for admissible features in _admissible_features
    public double[] _relevance; // variable importance for all predictors
    public double[] _relevance_valid; // equals to _relevance but may change in order
    public String[] _all_predictor_names;
    public String[] _all_predictor_names_valid;
    public DistributionFamily _distribution;
    public String _relevance_cmi_key;
    public String _relevance_cmi_key_valid;
    public String _relevance_cmi_key_xval;
    public Key<Frame> _relCmiKey;
    public Key<Frame> _relCmiKey_valid;
    public Key<Frame> _relCmiKey_xval;
    public String[] _topKFeatures;
    public long _validNonZeroNumRows;

    @Override
    public ModelCategory getModelCategory() {
      if (bernoulli.equals(_distribution)) {
        return ModelCategory.Binomial;
      } else if (multinomial.equals(_distribution)) {
        return ModelCategory.Multinomial;
      } else if (ordinal.equals(_distribution)) {
        return ModelCategory.Ordinal;
      }
      throw new IllegalArgumentException("Infogram currently only support binomial and multinomial classification");
    }

    public void setDistribution(DistributionFamily distribution) {
      _distribution = distribution;
    }
    
    public InfogramModelOutput(Infogram b) {
      super(b);
      if (glm.equals(b._parms._infogram_algorithm)) {
        if (binomial.equals(((GLMModel.GLMParameters) b._parms._infogram_algorithm_parameters)._family))
          _distribution = bernoulli;
        else if (multinomial.equals(((GLMModel.GLMParameters) b._parms._infogram_algorithm_parameters)._family))
          _distribution = multinomial;
        else if (ordinal.equals(((GLMModel.GLMParameters) b._parms._infogram_algorithm_parameters)._family))
          _distribution = ordinal;
      } else {
        _distribution = b._parms._infogram_algorithm_parameters._distribution;
      }
    }
    
    /***
     * Generate arrays containing only admissible features which are predictors with both cmi >= cmi_threshold and
     * relevance >= relevance_threshold
     * 
     * @param relCMIFrame H2O Frame containing relevance, cmi, ... info
     * @param validFrame true if validation dataset exists
     * @param cvFrame true if cross-validation is enabled
     */
    public void extractAdmissibleFeatures(Frame relCMIFrame, boolean validFrame, boolean cvFrame) {
      long numRow = relCMIFrame.numRows();
      // relCMIFrame contains c1:column, c2:admissible, c3:admissible_index, c4:relevance, c5:cmi, c6 cmi_raw
      List<Double> varimps = new ArrayList<>();
      List<Double> predictorCMI = new ArrayList<>();
      List<Double> predictorCMIRaw = new ArrayList<>();
      List<String> admissiblePred = new ArrayList<>();
      for (long rowIndex=0; rowIndex<numRow; rowIndex++) {
        if (relCMIFrame.vec(_ADMISSIBLE_PREDICTOR_INDEX).at(rowIndex) > 0) {
          varimps.add(relCMIFrame.vec(_RELEVANCE_INDEX).at(rowIndex));
          predictorCMI.add(relCMIFrame.vec(_CMI_INDEX).at(rowIndex));
          predictorCMIRaw.add(relCMIFrame.vec(_CMI_RAW_INDEX).at(rowIndex));
          admissiblePred.add(relCMIFrame.vec(_COLUMN_INDEX).stringAt(rowIndex));
        }
      } 
      if (validFrame) {
        _admissible_features_valid = admissiblePred.toArray(new String[admissiblePred.size()]);
      } else if (cvFrame) {
        _admissible_features_xval = admissiblePred.toArray(new String[admissiblePred.size()]);
      } else {
        _admissible_features = admissiblePred.toArray(new String[admissiblePred.size()]);
        _admissible_cmi = predictorCMI.stream().mapToDouble(i -> i).toArray();
        _admissible_cmi_raw = predictorCMIRaw.stream().mapToDouble(i -> i).toArray();
        _admissible_relevance = varimps.stream().mapToDouble(i -> i).toArray();
      }
    }
    
    public Key<Frame> setCMIRelFrame(boolean validPresent, boolean buildCore) {
      Frame cmiRelFrame = generateCMIRelevance(_all_predictor_names, _admissible, _admissible_index, _relevance, _cmi,
              _cmi_raw, buildCore);
      _relCmiKey = cmiRelFrame._key;
      _relevance_cmi_key = _relCmiKey.toString();
      if (validPresent) {  // generate relevanceCMI frame for validation dataset
        Frame cmiRelFrameValid = generateCMIRelevance(_all_predictor_names_valid, _admissible_valid, _admissible_index_valid, 
                _relevance_valid, _cmi_valid, _cmi_raw_valid, buildCore);
        _relCmiKey_valid = cmiRelFrameValid._key;
        _relevance_cmi_key_valid = _relCmiKey_valid.toString();
      }
      return cmiRelFrame._key;
    }

    /**
     * Copy over info to model._output for _cmi_raw, _cmi, _topKFeatures,
     * _all_predictor_names.  Derive _admissible for predictors if cmi >= cmi_threshold and 
     * relevance >= relevance_threshold.  Derive _admissible_index as distance from point with cmi = 1 and 
     * relevance = 1.  In addition, all arrays are sorted on _admissible_index.
     * 
     * @param cmiRaw
     * @param cmi
     * @param topKPredictors
     * @param varImp
     * @param cmiThreshold
     * @param relThreshold
     */
    public void copyCMIRelevance(double[] cmiRaw, double[] cmi, String[] topKPredictors,
                                  TwoDimTable varImp, double cmiThreshold, double relThreshold) {
      _cmi_raw = new double[cmi.length];
      System.arraycopy(cmiRaw, 0, _cmi_raw, 0, _cmi_raw.length);
      _admissible_index = new double[cmi.length];
      _admissible = new double[cmi.length];
      _cmi = cmi.clone();
      _topKFeatures = topKPredictors.clone();
      _all_predictor_names = topKPredictors.clone();
      int numRows = varImp.getRowDim();
      String[] varRowHeaders = varImp.getRowHeaders();
      List<String> relNames = new ArrayList<String>(Arrays.asList(varRowHeaders));
      _relevance = new double[numRows];
      copyRelGenerateAdmissibleIndex(numRows, cmiThreshold, relThreshold, relNames, varImp, _cmi, _cmi_raw, _relevance,
              _admissible_index, _admissible, _all_predictor_names);
    }
    
    public void copyRelGenerateAdmissibleIndex(int numRows, double cmiThreshold, double relThreshold, 
                                               List<String> relNames, TwoDimTable varImp, double[] cmi, double[] cmi_raw,
                                               double[] relevance, double[] admissible_index, double[] admissible, 
                                               String[] all_predictor_names) {
      for (int index = 0; index < numRows; index++) { // extract predictor with varimp >= threshold
        int newIndex = relNames.indexOf(all_predictor_names[index]);
        relevance[index] = (double) varImp.get(newIndex, 1);
        double temp1 = 1-relevance[index];
        double temp2 = 1-cmi[index];
        admissible_index[index] =  Math.sqrt(temp1*temp1+temp2*temp2);
        admissible[index] = (relevance[index] >= relThreshold && cmi[index] >= cmiThreshold) ? 1 : 0;
      }
      int[] indices = IntStream.range(0, cmi.length).toArray();
      sort(indices, admissible_index, -1, 1);
      sortCMIRel(indices, relevance, cmi_raw, cmi, all_predictor_names, admissible_index, admissible);
    }

    public void copyCMIRelevanceValid(double[] cmiRaw, double[] cmi, TwoDimTable varImp, double cmiThreshold,
                                      double relThreshold) {
      _cmi_raw_valid = new double[cmi.length];
      System.arraycopy(cmiRaw, 0, _cmi_raw_valid, 0, _cmi_raw_valid.length);
      _admissible_index_valid = new double[cmi.length];
      _admissible_valid = new double[cmi.length];
      _cmi_valid = cmi.clone();
      int numRows = varImp.getRowDim();
      String[] varRowHeaders = varImp.getRowHeaders();
      List<String> relNames = new ArrayList<String>(Arrays.asList(varRowHeaders));
      _all_predictor_names_valid = _topKFeatures.clone();
      _relevance_valid = new double[numRows];
      copyRelGenerateAdmissibleIndex(numRows, cmiThreshold, relThreshold, relNames, varImp, _cmi_valid, _cmi_raw_valid,
              _relevance_valid, _admissible_index_valid, _admissible_valid, _all_predictor_names_valid);
    }

    /***
     * This method will sort _relvance, _cmi_raw, _cmi_normalize, _all_predictor_names such that features that
     * are closest to upper right corner of infogram comes first with the order specified in the index
     * @param indices
     */
    public static void sortCMIRel(int[] indices, double[] relevance, double[] cmiRawA, double[] cmi, 
                            String[] allPredictorNames, double[] admissibleIndex, double[] admissibleA) {
      int indexLength = indices.length;
      double[] rel = new double[indexLength];
      double[] cmiRaw = new double[indexLength];
      double[] cmiNorm = new double[indexLength];
      double[] distanceCorner = new double[indexLength];
      String[] predNames = new String[indexLength];
      double[] admissible = new double[indexLength];
        for (int index = 0; index < indexLength; index++) {
          rel[index] = relevance[indices[index]];
          cmiRaw[index] = cmiRawA[indices[index]];
          cmiNorm[index] = cmi[indices[index]];
          predNames[index] = allPredictorNames[indices[index]];
          distanceCorner[index] = admissibleIndex[indices[index]];
          admissible[index] = admissibleA[indices[index]];
        }
        System.arraycopy(rel, 0, relevance, 0, indexLength);
        System.arraycopy(cmiNorm, 0, cmi, 0, indexLength);
        System.arraycopy(cmiRaw, 0, cmiRawA, 0, indexLength);
        System.arraycopy(predNames, 0, allPredictorNames, 0, indexLength);
        System.arraycopy(distanceCorner, 0, admissibleIndex, 0, indexLength);
        System.arraycopy(admissible, 0, admissibleA, 0, indexLength);
    }
  }

  @Override
  public boolean haveMojo() {
     return false;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  protected Futures remove_impl(Futures fs, boolean cascade) {
    super.remove_impl(fs, cascade);
    Keyed.remove(_output._relCmiKey, fs, true);
    Keyed.remove(_output._relCmiKey_valid, fs, true);
    Keyed.remove(_output._relCmiKey_xval, fs, true);
    return fs;
  }

  @Override
  protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    if (_output._relCmiKey != null)
      ab.putKey(_output._relCmiKey);
    if (_output._relCmiKey_valid != null)
      ab.putKey(_output._relCmiKey_valid);
    if (_output._relCmiKey_xval != null)
      ab.putKey(_output._relCmiKey_xval);
    return super.writeAll_impl(ab);
  }

  @Override
  protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}
