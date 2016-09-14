#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{



  // Flags


  // Variables
  int i, j;
  Type jnll=0;


  //////////////////////////////////////////////////
  ////////// Observation Model: Catch Data /////////
  //////////////////////////////////////////////////

  // Data
  DATA_MATRIX(N_catched);          // Data matrix of dimensions n_years x n_populations
  DATA_MATRIX(logit_P_catch_mean); // Prior matrix of dimensions n_years x n_populations
  DATA_MATRIX(logit_P_catch_SE);   // Prior matrix of dimensions n_years x n_populations

  // Parameters
  // Random effects
  PARAMETER_MATRIX(logit_P_catch);       // Probability of being catched, matrix of dimensions n_years x n_populations

  int n_pop = N_catched.cols();
  int n_yrs = N_catched.rows();

  // Probability of being catched
        // It is possible to make a more complicated model where logit_P_catch varies //
        // among populations and within populations (among years).                    //
        // The benefit of this is that it is possible to have NA                      //
  for(j=0;j<n_pop;j++){
    for(i=0;i<n_yrs;i++){
      jnll -= dnorm(logit_P_catch(i,j), logit_P_catch_mean(i,j), logit_P_catch_SE(i,j), true);
    }
  }

  // Inverse logit transformation to get P_catch_true
  matrix<Type> P_catch(n_yrs, n_pop);
  for(j=0;j<n_pop;j++){
    for(i=0;i<n_yrs;i++){
      P_catch(i, j) = exp(logit_P_catch(i,j))/(1.0+exp(logit_P_catch(i,j)));
    }
  }

  // Number of adults
  matrix<Type> N_adults_observed(n_yrs, n_pop);
  matrix<Type> N_spawners_observed(n_yrs, n_pop);
  for(j=0;j<n_pop;j++){
    for(i=0;i<n_yrs;i++){
      N_adults_observed(i,j) = N_catched(i,j) * (1/P_catch(i,j));
      N_spawners_observed(i,j) = N_adults_observed(i,j) - N_catched(i,j);
    }
  }

  /////////////////////////////////
  //////// Process model //////////
  /////////////////////////////////

  // so far this is just a simple model with non-overlapping generations
  // Parameters
  // Fixed effects (means)
  PARAMETER(r_global_mean); // Global mean of r

  // Fixed effects (variances)
  PARAMETER(lnSD_r_pop); // SD among population
  PARAMETER(lnSD_r_yrs); // SD among years (same for all populaitons)
  PARAMETER(lnSD_r_yrs_wPop); // SD among years (unique effect for each population)
  PARAMETER(lnSigma); // Residual variances

  // Random effects
  PARAMETER_MATRIX(ln_N_est);
  PARAMETER_VECTOR(r_pop);        // stable pop differences in r
  PARAMETER_VECTOR(r_yrs);        // Common effect of year on r, dimension n_yrs - 1

  for(j=0;j<n_pop;j++){
    for(i=1;i<n_yrs;i++){
      Type expectation = ln_N_est(i-1, j) + r_global_mean + r_pop(j) + r_yrs(i-1);
      jnll -= dnorm(ln_N_est(i, j), expectation, exp(lnSD_r_yrs_wPop), true);
    }
  }

  for(j=0;j<n_pop;j++){
    jnll -= dnorm(r_pop(j), Type(0.0), exp(lnSD_r_pop), true);
  }

  for(i=1;i<n_yrs;i++){
    jnll -= dnorm(r_yrs(i-1), Type(0.0), exp(lnSD_r_yrs), true);
  }


  // Residual log likelihood
  for(j=0;j<n_pop;j++){
    for(i=0;i<n_yrs;i++){
      jnll -= dnorm(log(N_adults_observed(i, j)), ln_N_est(i, j), exp(lnSigma), true);
    }
  }



  return jnll;
}



