functions{
  vector human_choice_rng(int s, vector S, real a, real b){
    vector[s] out; // holds predictions for the human coding
    for (i in 1:s) {
      real skill = beta_rng(a, b);
      if (S[i] == 1) {
        out[i] = binomial_rng(1, skill);
      } else {
        out[i] = binomial_rng(1, 1 - skill);
      }
    }
    return out;
  }
  vector inverse_logit(int s, vector S){
    vector[s] out;
    for (i in 1:s){
      out[i] = exp(S[i]) / (1 + exp(S[i]));
    }
    return out;
  }
  real inv_l(int x) {
    real out;
    out = inv_logit(x);
    return out;
  }
  matrix generate_normal_variate_rng(int n, int m, vector mu, matrix Sigma) {

    matrix[n, m + 1] out;
    vector[m] betas;
    real alpha;
    betas = [.5, .2, .6]';
    alpha = 0;
    for (i in 1:n){
      real true_p;
      real mu_1;
      row_vector[m] multinorm;
      multinorm = multi_normal_rng(mu, Sigma)';
      // mu_1 = alpha;
      // for (k in 1:m) {
      //   mu_1 = mu_1 + betas[k]*multinorm[k];
      // }
      // mu_1 = alpha + .5*multinorm[1] + .2 * multinorm[2] + .6*multinorm[3];
      mu_1 = alpha + multinorm * betas;
      true_p = inv_logit(mu_1);
      // true_p    = logistic_cdf(0 + .5*multinorm[1] + .2 * multinorm[2] + .6*multinorm[3], 0 , 1);
      out[i, ]  = append_col(binomial_rng(1, true_p), multinorm);
    }
    return out;
  }
}



