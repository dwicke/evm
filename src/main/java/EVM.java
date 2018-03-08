import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.*;
import java.util.stream.Collectors;

public class EVM {

    final EuclideanDistance euclideanDistance = new EuclideanDistance();
    final Weibull weibull = new Weibull();

    class Model {
        List<Weibull.WeibullParams> psi_l;
        List<Integer> indices;

        public Model(int label, List<Weibull.WeibullParams> psi_l, List<Integer> indices) {
            this.indices = indices;
            this.psi_l = psi_l;
        }
    }


    public void train(Map<Integer, double[][]> X, int y[], int tau, double sigma, int labels[], int numSamples) {

        List<Model> perClassModel = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {

            List<Weibull.WeibullParams> psi_l = fit(X, y, tau, labels[i], numSamples);
            List<Integer> indices = reduce(X.get(labels[i]), psi_l, sigma);

            perClassModel.add(new Model(labels[i], psi_l, indices ));
        }
    }


    public int predict(List<Model> model, double x[], int labels[], double threshold) {
        return 0;
    }

    public double psi(Weibull.WeibullParams params, double x[], double x_prime[]) {
        return 3;
    }


    public List<Weibull.WeibullParams> fit(Map<Integer, double[][]> X, int y[], int tau, int label, int numSamples) {


        // create the distance matrix
        // this could probably be greatly improved
        double[][] distMatrix = new double[X.get(label).length][numSamples - X.get(label).length];
        double[][] inclass = X.get(label);
        for (int i = 0; i < inclass.length; i++) { // for each
            for (Map.Entry<Integer, double[][]> someClass : X.entrySet()) {
                if (someClass.getKey() != label) {
                    double[][] outclass = someClass.getValue();
                    for (int j = 0; j < outclass.length; j++) {
                        distMatrix[i][j] = euclideanDistance.compute(inclass[i], outclass[j]);
                    }
                }
            }
        }

        List<Weibull.WeibullParams> evs = new ArrayList<>();
        // now create the EV's
        for (int i = 0; i < distMatrix.length; i++) {
            // Weibull fit low( 1/2 × sort(Di)[: τ ])
            evs.add(weibull.fit(Arrays.stream(distMatrix[i]).sorted().boxed().collect(Collectors.toList()).subList(0,tau).stream().map(val -> .5 * val).collect(Collectors.toList())));
        }

        return evs;
    }


    public List<Integer> reduce(double[][] X, List<Weibull.WeibullParams> psi_l, double sigma) {
        // corresponds to Set Cover Model Reduction





        return null;
    }


}
