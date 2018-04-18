package com.dwicke.evm;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class EVM {


    boolean COSINE = false;
    boolean EUCLID = false;
    boolean L1 = false;

    public class Model {
        List<WeibullParams> psi_l;
        int label;
        double xs[][];

        public Model(int label, List<WeibullParams> psi_l, double[][] examples) {
            this.label = label;
            // each row of examples corresponds to
            this.psi_l = psi_l;
            this.xs = examples;
        }
    }



    public EVM(boolean useCosine) {
        COSINE = useCosine;
    }

    /**
     *
     * @param X key is the class label and the value is an array of examples where each row is an example
     * @param tau is the number of features to use
     * @param sigma is the coverage threshold
     * @param labels is the list of labels for the data
     * @param numSamples the total number of examples
     */
    public List<Model> train(Map<Integer, double[][]> X, int tau, int labels[], int numSamples, double sigma, int maxEVs, double tolerance) {

        List<Model> perClassModel = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {

            List<WeibullParams> psi_l = fit(X, tau, labels[i], numSamples);
            List<Integer> indices = fixedSizeReduction(X.get(labels[i]), psi_l,maxEVs, tolerance);//reduce(X.get(labels[i]), psi_l, sigma);

            List<WeibullParams> reduced_psi_l = new ArrayList<>();
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


    /**
     * Predict the class label or return -1 for unknown
     * @param model the trained model
     * @param x the example to predict
     * @param threshold the threshold for rejecting
     * @return
     */
    public int predict(List<Model> model, double x[], double threshold) {

        int predicted = -1;
        double maxPsiValClass = -1;

        for (Model m : model) {

            double maxPsi = -1;
            int i = 0;
            for (WeibullParams params : m.psi_l) {
                double psiVal = psi(params, getDistance(m.xs[i], x));
                if (psiVal > maxPsi) {
                    maxPsi = psiVal;
                }
            }
            if (maxPsi > maxPsiValClass) {
                maxPsiValClass = maxPsi;
                predicted = m.label;
            }
        }
        if (maxPsiValClass > threshold) {
            return predicted;
        }
        return -1;
    }

    public double psi(WeibullParams params, double dist) {

        //System.err.println("e^ = ((" + (- dist / params.lam) + ") ^"+params.k + ") = " + Math.exp(Math.pow(- dist / params.lam, params.k)));
//        System.err.println("lam = " + params.lam + " k = " + params.k + " dist = " + dist);
        return Math.exp(Math.pow(- dist / params.lam, params.k));
    }


    public List<WeibullParams> fit(Map<Integer, double[][]> X, int tau, int label, int numSamples) {


        // create the distance matrix
        // this could probably be greatly improved
        double[][] distMatrix = new double[X.get(label).length][numSamples - X.get(label).length];
        double[][] inclass = X.get(label);

//        System.err.println("label = " + label);
//        System.err.println("Dist " + X.get(label).length + ", " + (numSamples - X.get(label).length));
//        System.err.println("inClass " + inclass.length);

        for (int i = 0; i < inclass.length; i++) { // for each
            int start = 0;
            for (Map.Entry<Integer, double[][]> someClass : X.entrySet()) {
                if (someClass.getKey() != label) {
                    //System.err.println(" some class label = " + someClass.getKey());
                    double[][] outclass = someClass.getValue();
                    for (int j = 0; j < outclass.length; j++) {
                        distMatrix[i][start] = getDistance(inclass[i], outclass[j]);
//                        System.err.println("Dist = " + distMatrix[i][start]);
                        start++;
                    }

                }
            }
        }

        List<WeibullParams> evs = new ArrayList<>();
        // now create the EV's
        for (int i = 0; i < distMatrix.length; i++) {
            // com.dwicke.evm.Weibull fit low( 1/2 × sort(Di)[: τ ])
            // first sort then remove the zeros
            // then

            List<Double> a = Arrays.stream(distMatrix[i]).sorted().dropWhile(n -> n == 0).boxed().collect(Collectors.toList()).subList(0,tau).stream().map(val -> .5 * val).collect(Collectors.toList());
            System.err.println(i + " Sorted = " + a.size() + ": ");
            for (double ab : a) {
                System.err.print(ab + ", ");
            }
            System.err.println("end sorted");

            evs.add(fit(Arrays.stream(distMatrix[i]).sorted().dropWhile(n -> n == 0).boxed().collect(Collectors.toList()).subList(0,tau).stream().map(val -> .5 * val).collect(Collectors.toList())));
        }

        return evs;
    }


    /**
     * Reduce the size of the model
     * @param X matrix corresponding to examples all within the same class (each row corresponds to a feature vector)
     * @param psi_l the weibull parameters for each feature vector
     * @param sigma the
     * @return
     */
    public List<Integer> reduce(double[][] X, List<WeibullParams> psi_l, double sigma) {
        // corresponds to Set Cover Model Reduction

        // So the idea is that we first generate a N_l x N_l pairwise distance matrix
        double[][] distMatrix = new double[X.length][X.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X.length; j++) {
                distMatrix[i][j] = getDistance(X[i], X[j]);
                //System.err.println("i = " + X[i].toString() +  " j = " + X[j].toString() + "dist matrix = " + distMatrix[i][j]);
            }
        }



        HashMap<Integer, Set<Integer>> S = new HashMap<>();
        Set<Integer> universe = new HashSet<>();

        for (int i = 0; i < X.length; i++) {
            universe.add(i);
            for (int j = 0; j < X.length; j++) {
                if (psi(psi_l.get(i), distMatrix[i][j]) >= sigma) {
                    Set<Integer> a = S.getOrDefault(i, new HashSet<>());
                    a.add(j);
                    S.putIfAbsent(i, a);
//                    System.err.println("Adding " + j);
                } else {
//                    System.err.println("Not adding " + j);
//                    System.err.println("psi = " + psi(psi_l.get(i), distMatrix[i][j]));

                }
            }
        }


        List<Integer> indices = new ArrayList<>();
        Set<Integer> C = new HashSet<>();


        // now do greedy set cover
        while(!C.containsAll(universe)) {
            Set<Integer> maxS = null;
            Integer index = -1;
            int maxDif = -1;
            for (Map.Entry<Integer, Set<Integer>> e : S.entrySet()) {
                int dif = difference(e.getValue(), C).size();
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


    public Set<Integer> difference(Set<Integer> a, Set<Integer> b) {

        Set<Integer> diff = new HashSet<>();

        for (Integer i : a) {
            diff.add(i);
        }

        // a - b
        for (Integer i : a) {
            if (b.contains(i)) {
                diff.remove(i);
            }
        }

        return diff;
    }


    public List<Integer> fixedSizeReduction(double[][] X, List<WeibullParams> psi_l, int maxEVs, double tolerance) {
        double sigma_min = 0.0;
        double sigma_old = 1.0;
        double sigma_max = 1.0;
        while(true) {

            double sigma = (sigma_min + sigma_max) / 2.0;
            List<Integer> I = reduce(X, psi_l, sigma);
            int M = I.size();
            boolean Cfour = M == maxEVs || Math.abs(sigma - sigma_old) <= tolerance;
            if (X.length - maxEVs >= M - maxEVs &&  M - maxEVs > 0) {
                sigma_max = sigma;
            } else if ((X.length - maxEVs >= M - maxEVs &&  M - maxEVs < 0) || (X.length - maxEVs < M - maxEVs)) {
                sigma_min = sigma;
            }

            if (Cfour) {
                return I.stream().collect(Collectors.toList()).subList(0,maxEVs);
            }


        }
    }

    public double getDistance(double[] a, double[] b) {
        if (COSINE) {
            return cosineSimilarity(a, b);
        }else if (EUCLID){
            return euclideanDist(a, b);
        } else if (L1){
            return LoneDist(a, b);
        } else {
            return jacardDist(a, b);
        }
    }

    public static double jacardDist(double[] vectorA, double[] vectorB) {

        double num = 0;
        double den = 0;

        for (int i = 0; i < vectorA.length; i++) {

            if (vectorA[i] > 0 || vectorB[i] > 0) {
                den++;
                if (vectorA[i] > 0 && vectorB[i] > 0) {
                    num++;
                }
            }

        }
        return 1.0 - num / den;
    }

    public static double LoneDist(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += Math.abs(vectorA[i] - vectorB[i]);
        }
        return dotProduct;
    }

    public static double euclideanDist(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
        }
        return Math.sqrt(dotProduct);
    }

    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public class WeibullParams {
        public double k = 1.0;
        public double lam;
    }

    double eps = .000006;
    int iters = 100;

    /**
     * Fits a 2-parameter com.dwicke.evm.Weibull distribution to the given data using maximum-likelihood estimation.
     * each value in x must be > 0
     * based off python version here:
     * https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_numpy.py
     * @param x > 0 for all values
     * @return weibull parameters populated in a WeibullParams object
     */
    public WeibullParams fit(List<Double> x) {

//        System.err.println("fitting");
//        for (Double d : x) {
//
//                System.err.println(d);
//
//        }

        List<Double> ln_x = x.stream().map(val -> Math.log(val)).collect(Collectors.toList());
        double mean_ln_x = ln_x.stream().mapToDouble(Double::doubleValue).sum() / (double) ln_x.size();
        WeibullParams params = new WeibullParams();
        double kPlusOne = params.k;

        for (int i = 0; i < iters; i++) {
            List<Double> x_k = x.stream().map(val -> Math.pow(val, params.k)).collect(Collectors.toList());
            List<Double> x_k_ln_x = IntStream.range(0, Math.min(x_k.size(), ln_x.size())).mapToDouble(p -> x_k.get(p) * ln_x.get(p)).boxed().collect(Collectors.toList());
            double ff = x_k_ln_x.stream().mapToDouble(Double::doubleValue).sum();
            double fg = x_k.stream().mapToDouble(Double::doubleValue).sum();
            double f = ff / fg - mean_ln_x - (1.0 / params.k);

            // Calculate second derivative d^2f/dk^2
            double ff_prime = IntStream.range(0, Math.min(x_k_ln_x.size(), ln_x.size())).mapToDouble(p -> x_k_ln_x.get(p) * ln_x.get(p)).sum();
            double fg_prime = ff;
            double f_prime = (ff_prime / fg - (ff/fg * fg_prime/fg)) + (1.0 / (params.k * params.k));

            // Newton-Raphson method k = k - f(k;x)/f'(k;x)
            params.k -= f / f_prime;

            if (Math.abs(params.k - kPlusOne) < eps) {
                break;
            }
            kPlusOne = params.k;
        }
        double x_k_avg = x.stream().mapToDouble(val -> Math.pow(val,params.k)).sum() / (double) x.size();
        params.lam = Math.pow(x_k_avg, (1.0 / params.k));
        return params;
    }

}

