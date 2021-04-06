package hex.util;

import hex.genmodel.utils.DistributionFamily;
import hex.glm.GLMModel;
import water.exceptions.H2OIllegalArgumentException;

import static hex.genmodel.utils.DistributionFamily.bernoulli;
import static hex.glm.GLMModel.GLMParameters.Family.*;

public class DistributionUtils {
    public static DistributionFamily familyToDistribution(GLMModel.GLMParameters.Family aFamily) {
        if (aFamily == GLMModel.GLMParameters.Family.binomial) {
            return bernoulli;
        }
        try {
            return Enum.valueOf(DistributionFamily.class, aFamily.toString());
        }
        catch (IllegalArgumentException e) {
            throw new H2OIllegalArgumentException("Don't know how to find the right DistributionFamily for Family: " + aFamily);
        }
    }
    
    public static GLMModel.GLMParameters.Family distributionToFamily(DistributionFamily distribution) {
        if (bernoulli.equals(distribution))
            return binomial;
        try {
            return Enum.valueOf(GLMModel.GLMParameters.Family.class, distribution.toString());
        } catch (IllegalArgumentException e) {
            throw new H2OIllegalArgumentException("Don't know how to find the right Family for DistributionFamily: " + distribution);
        }
    }
}
