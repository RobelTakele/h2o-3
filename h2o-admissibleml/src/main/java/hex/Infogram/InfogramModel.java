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
import water.fvec.Vec;
import water.udf.CFuncRef;
import water.util.TwoDimTable;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.IntStream;

import static hex.Infogram.InfogramModel.InfogramParameters.Algorithm.glm;
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
    public double _total_information_threshold = -1;  // relevance threshold for core infogram
    public double _net_information_threshold = -1;    // cmi threshold for core infogram
    public double _safety_index_threshold = -1;       // cmi threshold for safe infogram
    public double _relevance_index_threshold = -1;    // relevance threshold for safe infogram
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
    public double[] _admissible_cmi;  // conditional info for admissible features in _admissible_features
    public double[] _admissible_cmi_raw;  // conditional info for admissible features in _admissible_features raw
    public double[] _admissible_relevance;  // varimp values for admissible features in _admissible_features
    public String[] _admissible_features; // predictors chosen that exceeds both conditional_info and varimp thresholds
    public double[] _admissible_index;  // store distance from 1,1 corner of infogram plot
    public double[] _admissible; // 0 if predictor is admissible and 1 otherwise
    public DistributionFamily _distribution;
    public double[] _cmi_raw; // cmi before normalization and for all predictors
    public double[] _cmi; // normalized cmi
    public String[] _all_predictor_names;
    public double[] _relevance; // variable importance for all predictors
    public String _relevance_cmi_key;
    public Key<Frame> _relCmiKey;
    public String[] _topKFeatures;

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
     * @param varImp
     * @param topKPredictors
     * @param cmi
     * @param cmiRaw
     * @param cmiThreshold
     * @param varImpThreshold
     */
    public void extractAdmissibleFeatures(TwoDimTable varImp, String[] topKPredictors, double[] cmi, double[] cmiRaw,
                                          double cmiThreshold, double varImpThreshold) {
      int numRows = varImp.getRowDim();
      List<Double> varimps = new ArrayList<>();
      List<Double> predictorCMI = new ArrayList<>();
      List<Double> predictorCMIRaw = new ArrayList<>();
      List<String> topKList = new ArrayList<>(Arrays.asList(topKPredictors));
      List<String> admissiblePred = new ArrayList<>();
      String[] varRowHeaders = varImp.getRowHeaders();
      for (int index = 0; index < numRows; index++) { // extract predictor with varimp >= threshold
        double varimp = (double) varImp.get(index, 1);
        if (varimp >= varImpThreshold) {
          int predIndex = topKList.indexOf(varRowHeaders[index]);
          if (cmi[predIndex] >= cmiThreshold) {
            varimps.add(varimp);
            predictorCMI.add(cmi[predIndex]);
            predictorCMIRaw.add(cmiRaw[predIndex]);
            admissiblePred.add(topKPredictors[predIndex]);
          }
        }
      }
      _admissible_features = admissiblePred.toArray(new String[admissiblePred.size()]);
      _admissible_cmi = predictorCMI.stream().mapToDouble(i -> i).toArray();
      _admissible_cmi_raw = predictorCMIRaw.stream().mapToDouble(i->i).toArray();
      _admissible_relevance = varimps.stream().mapToDouble(i -> i).toArray();
    }

    /***
     * Generate frame that contains information columns, admissible, admissible_index, relevance, cmi and cmi_raw.
     * Note that the frame is sorted on admissible_index from 0 to 1.
     * 
     * @return H2OFrame key
     */
    public Key<Frame> generateCMIRelFrame(boolean core) {
      Vec.VectorGroup vg = Vec.VectorGroup.VG_LEN1;
      Vec vName = Vec.makeVec(_all_predictor_names, vg.addVec());
      Vec vAdm = Vec.makeVec(_admissible, vg.addVec());
      Vec vAdmIndex = Vec.makeVec(_admissible_index, vg.addVec());
      Vec vRel = Vec.makeVec(_relevance, vg.addVec());
      Vec vCMI = Vec.makeVec(_cmi, vg.addVec());
      Vec vCMIRaw = Vec.makeVec(_cmi_raw, vg.addVec());
      String[] columnNames = core ? new String[]{"column", "admissible", "admissible_index", "total_information", 
              "net_information", "net_information_raw"} : new String[]{"column", "admissible", "admissible_index",
              "relevance_index", "safety_index", "safety_index_raw"};
      Frame cmiRelFrame = new Frame(Key.<Frame>make(), columnNames, new Vec[]{vName, vAdm, vAdmIndex, vRel, vCMI, vCMIRaw});
      DKV.put(cmiRelFrame);
      _relCmiKey = cmiRelFrame._key;
      _relevance_cmi_key = _relCmiKey.toString();
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
    public void copyCMIRelevance( double[] cmiRaw, double[] cmi, String[] topKPredictors,
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
      for (int index = 0; index < numRows; index++) { // extract predictor with varimp >= threshold
        int newIndex = relNames.indexOf(_all_predictor_names[index]);
        _relevance[index] = (double) varImp.get(newIndex, 1);
        double relDiff = 1-_relevance[index];
        double cmiDiff = 1-cmi[index];
        _admissible_index[index] = Math.sqrt(relDiff*relDiff+cmiDiff*cmiDiff);
        _admissible[index] = (_relevance[index] >= relThreshold && _cmi[index] >= cmiThreshold) ? 1 : 0;
      }
      int[] indices = IntStream.range(0, cmi.length).toArray();
      sort(indices, _admissible_index, -1, 1);
      sortCMIRel(indices);
    }

    /***
     * This method will sort _relvance, _cmi_raw, _cmi_normalize, _all_predictor_names such that features that
     * are closest to upper right corner of infogram comes first with the order specified in the index
     * @param indices
     */
    private void sortCMIRel(int[] indices) {
      int indexLength = indices.length;
      double[] rel = new double[indexLength];
      double[] cmiRaw = new double[indexLength];
      double[] cmiNorm = new double[indexLength];
      double[] distanceCorner = new double[indexLength];
      String[] predNames = new String[indexLength];
      double[] admissible = new double[indexLength];
      for (int index = 0; index < indexLength; index++) {
        rel[index] = _relevance[indices[index]];
        cmiRaw[index] = _cmi_raw[indices[index]];
        cmiNorm[index] = _cmi[indices[index]];
        predNames[index] = _all_predictor_names[indices[index]];
        distanceCorner[index] = _admissible_index[indices[index]];
        admissible[index] = _admissible[indices[index]];
      }
      _relevance = rel;
      _cmi = cmiNorm;
      _cmi_raw = cmiRaw;
      _all_predictor_names = predNames;
      _admissible_index = distanceCorner;
      _admissible = admissible;
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
    return fs;
  }

  @Override
  protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    if (_output._relCmiKey != null)
      ab.putKey(_output._relCmiKey);
    return super.writeAll_impl(ab);
  }

  @Override
  protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}
