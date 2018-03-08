import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Weibull {


    public class WeibullParams {
        public double k = 1.0;
        public double lam;
    }

    double eps = .000006;
    int iters = 100;

    /**
     * Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
     * each value in x must be > 0
     * based off python version here:
     * https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_numpy.py
     * @param x > 0 for all values
     * @return weibull parameters populated in a WeibullParams object
     */
    public WeibullParams fit(List<Double> x) {
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

