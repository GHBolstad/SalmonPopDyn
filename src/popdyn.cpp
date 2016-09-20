#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_INTEGER(n_t);    //maybe change this to start year and end year?
  DATA_INTEGER(n_pop);
  int n_class = 1;      //number of sea age categories in the model

  // Latent variables:
  PARAMETER_ARRAY(lnN_ad); //Number of adult female spawners

  // Flags


  // Variables
  int t, i, j;
  Type jnll=0;

  // exp tranformation
  array<Type> N_ad(n_t, n_pop, n_class);
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      for(j=0;j<n_class;j++){
        N_ad(t,i,j) = exp(lnN_ad(t,i,j));
      }}}



  //////////////////////////////////////////////////
  ////////// Observation Model: Catch Data /////////
  //////////////////////////////////////////////////

  // Data
  DATA_ARRAY(Catch_data_N);          // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(Catch_data_Kg);         // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(prop_fem);              // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(logit_P_catch_mean);    // Prior array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(logit_P_catch_SE);      // Prior array of dimensions n_years x n_populations x n_weight_class
  //DATA_ARRAY(logit_P_SeaAge1_mean);   // Prior for probability of a fish to have sea age 1
  //DATA_ARRAY(logit_P_SeaAge1_SE);
  //DATA_ARRAY(logit_P_SeaAge2_mean);   // Prior for probability of a fish to have sea age 2
  //DATA_ARRAY(logit_P_SeaAge2_SE);

  // Parameters
  PARAMETER(lnSigma_lnN); // Residual variance

  int n_class_data = 3;      //number of sea age categories in the data

  // Random effects
  //PARAMETER_ARRAY(logit_P_catch);       // Probability of being catched, array of dimensions n_years x n_populations x n_weight_class
  array<Type> logit_P_catch(n_t, n_pop, n_class_data);


  // Probability of being catched
        // It is possible to make a more complicated model where logit_P_catch varies //
        // among populations and within populations (among years).                    //
        // The benefit of this is that it is possible to have NA                      //
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      for(j=0;j<n_class_data;j++){
        //jnll -= dnorm(logit_P_catch(t, i, j), logit_P_catch_mean(t, i, j), logit_P_catch_SE(t, i, j), true);
        logit_P_catch(t, i, j) = logit_P_catch_mean(t, i, j);
      }}}


  // Inverse logit transformation to get P_catch_true
  array<Type> P_catch(n_t, n_pop, n_class_data);
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      for(j=0;j<n_class_data;j++){
        P_catch(t,i,j) = exp(logit_P_catch(t,i,j))/(1.0+exp(logit_P_catch(t,i,j)));
      }}}


  // Number of adults
  array<Type> N_female_adults_observed(n_t, n_pop, n_class_data);
  array<Type> N_female_spawners_observed(n_t, n_pop, n_class_data);
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      for(j=0;j<n_class_data;j++){
        N_female_adults_observed(t,i,j) = Catch_data_N(t,i,j) * (1/P_catch(t,i,j)) * prop_fem(t,i,j);
        N_female_spawners_observed(t,i,j) = Catch_data_N(t,i,j);//N_female_adults_observed(t,i,j) - Catch_data_N(t,i,j) * prop_fem(t,i,j);
      }}}

/*
  // Number of egg from the average weight of the adults
  array<Type> mean_kg_observed(n_t, n_pop, n_class);
  matrix<Type> N_egg(n_t, n_pop);
  N_egg.fill(0.0); // I THINK THIS WORKS...
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      mean_kg_observed(t,i,0) = Catch_data_Kg(t,i,0) / Catch_data_N(t,i,0);
      mean_kg_observed(t,i,1) = (Catch_data_Kg(t,i,1)+Catch_data_Kg(t,i,2)) / (Catch_data_N(t,i,1) + Catch_data_N(t,i,2));
      N_egg(t,i) += mean_kg_observed(t,i,0)*Type(1450.0)*(Type(1.0)*N_ad(t,i,0)+Type(0.0)*N_ad(t,i,1));
      N_egg(t,i) += mean_kg_observed(t,i,1)*Type(1450.0)*(Type(0.0)*N_ad(t,i,0)+Type(1.0)*N_ad(t,i,1));
    }}
*/

  // Residual log likelihood           // TODO: include a residual for mean kg?
  // NB! so far I use constants to attribute weight classes to sea age classes, TODO: use priors instead and maybe use data
  // I also add 1.0 to avid log transformation of zeros, I think I can skip tis, or use GLM!
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      jnll -= dnorm(log(N_female_spawners_observed(t, i, 0) + Type(1.0)), lnN_ad(t,i,0), exp(lnSigma_lnN), true);
      //jnll -= dnorm(log(N_female_spawners_observed(t, i, 1) + Type(1.0)), lnN_ad(t,i,1), exp(lnSigma_lnN), true);
      //jnll -= dnorm(log(N_female_spawners_observed(t, i, 0) + Type(1.0)), log(Type(1.0)*N_ad(t,i,0) + Type(0.0)*N_ad(t,i,1) + Type(1.0)), exp(lnSigma_lnN), true);
      //jnll -= dnorm(log(N_female_spawners_observed(t, i, 1) + N_female_spawners_observed(t, i, 2) + Type(1.0)), log(Type(0.0)*N_ad(t,i,0) + Type(1.0)*N_ad(t,i,1) + Type(1.0)), exp(lnSigma_lnN), true);
    }}




  /////////////////////////////////
  //////// Process model //////////
  /////////////////////////////////

  PARAMETER(lnSD_lnN_add_residual);
  PARAMETER(logitP_sea0Tad_mean);        Type P_sea0Tad_mean   = exp(logitP_sea0Tad_mean)/(1+exp(logitP_sea0Tad_mean));
  //PARAMETER(logitP_sea1Tad_mean);        Type P_sea1Tad_mean   = exp(logitP_sea1Tad_mean)/(1+exp(logitP_sea1Tad_mean));


  Type expectation=0;

  for(t=1;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      // Sea age 1:
      expectation = N_ad(t-1,i,0) * P_sea0Tad_mean;  // * P_sea0Tad_t(t-1) * P_sea0Tad_pop(i) ;
      jnll -= dnorm(lnN_ad(t,i,0), log(expectation), exp(lnSD_lnN_add_residual), true);

      // Sea age 2:
      //N_ad(t,i,1) = N_ad(t-1,i,1) * P_sea1Tad_mean;  // * P_sea1Tad_t(t-1) * P_sea1Tad_pop(i);
      //jnll -= dnorm(lnN_ad(t,i,1), log(expectation), exp(lnSD_lnN_add_residual), true);
    }}


  /*
  for(t=1;t<n_t;i++){
    jnll -= dnorm(logitP_sea0Tad_t(t-1), Type(0.0), exp(lnSD_r_yrs), true);
  }

  for(j=0;j<n_pop;j++){
    jnll -= dnorm(r_pop(j), Type(0.0), exp(lnSD_r_pop), true);
  }
  */



/*
  ///////// Parameters ////////
  // Fixed effects (means)
  PARAMETER(logitP_eggTriv_mean);        Type P_eggTriv_mean   = exp(logitP_eggTriv_mean)/(1+exp(logitP_eggTriv_mean));
  PARAMETER(logitP_rivTriv_mean);        Type P_rivTriv_mean   = exp(logitP_rivTriv_mean)/(1+exp(logitP_rivTriv_mean));
  PARAMETER(logitP_rivTsea0_mean);       Type P_rivTsea0_mean  = exp(logitP_rivTsea0_mean)/(1+exp(logitP_rivTsea0_mean));
  PARAMETER(logitP_sea0Tsea1_mean);      Type P_sea0Tsea1_mean = exp(logitP_sea0Tsea1_mean)/(1+exp(logitP_sea0Tsea1_mean));
  //PARAMETER(logitP_sea1Tsea1_mean);      Type P_sea1Tsea1_mean = exp(logitP_sea1Tsea1_mean)/(1+exp(logitP_sea1Tsea1_mean));
  PARAMETER(logitP_sea0Tad_mean);        Type P_sea0Tad_mean   = exp(logitP_sea0Tad_mean)/(1+exp(logitP_sea0Tad_mean));
  PARAMETER(logitP_sea1Tad_mean);        Type P_sea1Tad_mean   = exp(logitP_sea1Tad_mean)/(1+exp(logitP_sea1Tad_mean));
  PARAMETER(lnN_riv_start_mean);
  PARAMETER(lnN_sea0_start_mean);
  PARAMETER(lnN_sea1_start_mean);

  // Fixed effects (variances)
  //PARAMETER(lnSD_lnN_riv_residual);
  //PARAMETER(lnSD_lnN_sea0_residual);
  //PARAMETER(lnSD_lnN_sea1_residual);
  PARAMETER(lnSD_lnN_add_residual);
  PARAMETER(lnSD_lnN_riv_start);
  PARAMETER(lnSD_lnN_sea0_start);
  PARAMETER(lnSD_lnN_sea1_start);

  // Random effects
  //PARAMETER_MATRIX(lnN_riv);
  //PARAMETER_MATRIX(lnN_sea0);
  //PARAMETER_MATRIX(lnN_sea1);
  PARAMETER_VECTOR(lnN_riv_start);
  PARAMETER_VECTOR(lnN_sea0_start);
  PARAMETER_VECTOR(lnN_sea1_start);

  // declearing variables
  matrix<Type> N_riv(n_t, n_pop);
  matrix<Type> N_sea0(n_t, n_pop);
  matrix<Type> N_sea1(n_t, n_pop);

  // Initial population sizes
  for(i=0;i<n_pop;i++){
    jnll -= dnorm(lnN_riv_start(i), lnN_riv_start_mean, exp(lnSD_lnN_riv_start), true);
    jnll -= dnorm(lnN_sea0_start(i), lnN_sea0_start_mean, exp(lnSD_lnN_sea0_start), true);
    jnll -= dnorm(lnN_sea1_start(i), lnN_sea1_start_mean, exp(lnSD_lnN_sea1_start), true);
    N_riv(0,i) = exp(lnN_riv_start(i));
    N_sea0(0,i) = exp(lnN_sea0_start(i));
    N_sea1(0,i) = exp(lnN_sea1_start(i));
  }

  ///////////////  MODEL ///////////////
  Type expectation=0;
  for(t=1;t<n_t;t++){
    for(i=0;i<n_pop;i++){

          // River
          N_riv(t,i) = N_egg(t-1,i) * P_eggTriv_mean // * P_eggTriv_t(t-1) * P_eggTriv_pop(i)
                         + N_riv(t-1,i) * P_rivTriv_mean; // * P_rivTriv_t(t-1) * P_rivTriv_pop(i);
          //jnll -= dnorm(lnN_riv(t,i), log(expectation), exp(lnSD_lnN_riv_residual), true);

          // Sea first year
          N_sea0(t,i) = N_riv(t-1,i) * P_rivTsea0_mean; // * P_rivTsea0_t(t-1) * P_rivTsea0_pop(i);
          //jnll -= dnorm(lnN_sea0(t,i), log(expectation), exp(lnSD_lnN_sea0_residual), true);

          // Sea after first year
          N_sea1(t,i) = N_sea0(t-1,i) * P_sea0Tsea1_mean; // * P_sea1Tsea2_t(t-1) * P_sea1Tsea2_pop(i)
                         //+ N_sea1(t-1,i) * P_sea1Tsea1_mean; // * P_sea2Tsea2_t(t-1) * P_sea2Tsea2_pop(i);
          //jnll -= dnorm(lnN_sea2(t,i), log(expectation), exp(lnSD_lnN_sea2_residual), true);

          // Adult spawners //
          // Sea age 1:
          expectation = N_sea0(t-1,i) * P_sea0Tad_mean;  // * P_sea0Tad_t(t-1) * P_sea0Tad_pop(i);
          jnll -= dnorm(lnN_ad(t,i,0), log(expectation), exp(lnSD_lnN_add_residual), true);

          // Sea age 2:
          expectation = N_sea1(t-1,i) * P_sea1Tad_mean;  // * P_sea1Tad_t(t-1) * P_sea1Tad_pop(i);
          jnll -= dnorm(lnN_ad(t,i,1), log(expectation), exp(lnSD_lnN_add_residual), true);

        }}

*/

/*
  for(j=0;j<n_pop;j++){
    jnll -= dnorm(r_pop(j), Type(0.0), exp(lnSD_r_pop), true);
  }

  for(i=1;i<n_yrs;i++){
    jnll -= dnorm(r_yrs(i-1), Type(0.0), exp(lnSD_r_yrs), true);
  }

*/



  return jnll;
}



