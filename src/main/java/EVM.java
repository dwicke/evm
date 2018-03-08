import com.google.common.collect.Sets;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.*;
import java.util.stream.Collectors;

public class EVM {

    final EuclideanDistance euclideanDistance = new EuclideanDistance();
    final Weibull weibull = new Weibull();

    class Model {
        List<Weibull.WeibullParams> psi_l;
        int label;
        double xs[][];

        public Model(int label, List<Weibull.WeibullParams> psi_l, double[][] examples) {
            this.label = label;
            // each row of examples corresponds to
            this.psi_l = psi_l;
            this.xs = examples;
        }
    }

    /**
     *
     * @param X key is the class label and the value is an array of examples where each row is an example
     * @param tau is the number of features to use
     * @param sigma is the coverage threshold
     * @param labels is the list of labels for the data
     * @param numSamples the total number of examples
     */
    public List<Model> train(Map<Integer, double[][]> X, int tau, double sigma, int labels[], int numSamples) {

        List<Model> perClassModel = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {

            List<Weibull.WeibullParams> psi_l = fit(X, tau, labels[i], numSamples);
            List<Integer> indices = reduce(X.get(labels[i]), psi_l, sigma);

            List<Weibull.WeibullParams> reduced_psi_l = new ArrayList<>();
            double reduced_X_l[][] = new double[indices.size()][X.get(i)[0].length];
            int count = 0;
            for (Integer index : indices) {
                reduced_psi_l.add(psi_l.get(index));
                reduced_X_l[count] = X.get(i)[index];
            }

            perClassModel.add(new Model(labels[i], reduced_psi_l, reduced_X_l));

        }
        return perClassModel;
    }


    public int predict(List<Model> model, double x[], int label, double threshold) {

        int predicted = -1;
        double maxPsiValClass = -1;

        for (Model m : model) {

            double maxPsi = -1;
            int i = 0;
            for (Weibull.WeibullParams params : m.psi_l) {
                double psiVal = psi(params, euclideanDistance.compute(m.xs[i], x));
                if (psiVal > maxPsi) {
                    maxPsi = psiVal;
                }
            }
            if (maxPsi > maxPsiValClass) {
                maxPsiValClass = maxPsi;
                predicted = m.label;
            }
        }

        return predicted;
    }

    public double psi(Weibull.WeibullParams params, double dist) {
        return Math.exp(Math.pow(- dist / params.lam, params.k));
    }


    public List<Weibull.WeibullParams> fit(Map<Integer, double[][]> X, int tau, int label, int numSamples) {


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

        // So the idea is that we first generate a N_l x N_l pairwise distance matrix
        double[][] distMatrix = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                distMatrix[i][j] = euclideanDistance.compute(X[i], X[j]);
            }
        }



        HashMap<Integer, Set<Integer>> S = new HashMap<>();
        Set<Integer> universe = new HashSet<>();

        for (int i = 0; i < X.length; i++) {
            universe.add(i);
            for (int j = 0; j < X[i].length; j++) {
                if (psi(psi_l.get(i), distMatrix[i][j]) >= sigma) {
                    Set<Integer> a = S.getOrDefault(i, new HashSet<>());
                    a.add(j);
                }
            }
        }

        List<Integer> indices = new ArrayList<>();
        Set<Integer> C = new HashSet<>();

        //

        // now do greedy set cover
        // consider http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
        while(!C.containsAll(universe)) {
            Set<Integer> maxS = null;
            Integer index = -1;
            int maxDif = -1;
            for (Map.Entry<Integer, Set<Integer>> e : S.entrySet()) {
                int dif = Sets.difference(e.getValue(), C).size();
                if (dif > maxDif) {
                    index = e.getKey();
                    maxS = e.getValue();
                    maxDif = dif;
                }
            }

            C.addAll(maxS);
            indices.add(index);
            S.remove(index);
        }


        return indices;
    }


}
