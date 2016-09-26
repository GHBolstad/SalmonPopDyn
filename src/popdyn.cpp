#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_INTEGER(n_t);    //maybe change this to start year and end year?
  DATA_INTEGER(n_pop);
  int n_class = 2;      //number of sea age categories in the model

  // Latent variables:
  PARAMETER_ARRAY(lnN_ad); //Number of adult female spawners

  // Flags
  DATA_VECTOR(life_history);


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
  DATA_ARRAY(lnNfem_VR_data);        // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(lnNfem_SE_VR_data);     // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(ln_mean_kg_VR_data);    // Data array of dimensions n_years x n_populations x n_weight_class
  DATA_ARRAY(Nfem_catched_VR_data);  // !!!!! change this to log proporiton catched !!!!!!

  // not needed: int n_class_data = 3;      //number of sea age categories in the data

  // Priors for initial population sizes
  for(i=0;i<n_pop;i++){
    jnll -= dnorm(lnN_ad(0,i,0), lnNfem_VR_data(0, i, 0), lnNfem_SE_VR_data(0,i,0)/Type(100.0), true);
    if(life_history(i) != 1){
      jnll -= dnorm(lnN_ad(0,i,1), lnNfem_VR_data(0, i, 1), lnNfem_SE_VR_data(0,i,1)/Type(100.0), true);
      jnll -= dnorm(lnN_ad(1,i,1), lnNfem_VR_data(1, i, 1), lnNfem_SE_VR_data(1,i,1)/Type(100.0), true);
    }
  }

  // Observation model for Nfem
  for(t=1;t<n_t;t++){
    for(i=0;i<n_pop;i++){
        jnll -= dnorm(lnNfem_VR_data(t, i, 0), lnN_ad(t,i,0), lnNfem_SE_VR_data(t,i,0), true);
      }}

    for(t=2;t<n_t;t++){
      for(i=0;i<n_pop;i++){
        if(life_history(i) != 1){
          jnll -= dnorm(lnNfem_VR_data(t, i, 1), lnN_ad(t,i,1), lnNfem_SE_VR_data(t,i,1), true);
        }}}



  // Observation model for mean_kg
  // So far I just use the raw estimates, but I shuld be able to get SE on these, based on P and N of each weight class
  array<Type> ln_mean_kg = ln_mean_kg_VR_data;

  // Calculating number of egg
  matrix<Type> lnN_egg(n_t, n_pop);
  matrix<Type> N_egg(n_t, n_pop);
  N_egg.fill(0.0);
  for(t=0;t<n_t;t++){
    for(i=0;i<n_pop;i++){
      N_egg(t,i) += exp(ln_mean_kg_VR_data(t,i,0)+log(Type(1450.0))+lnN_ad(t,i,0));
      if(life_history(i) != 1){
        N_egg(t,i) += exp(ln_mean_kg_VR_data(t,i,1)+log(Type(1450.0))+lnN_ad(t,i,1));
      }
      //N_egg(t,i) += mean_kg_observed(t,i,0)*Type(1450.0)*(Type(1.0)*N_ad(t,i,0)+Type(0.0)*N_ad(t,i,1));
      //N_egg(t,i) += mean_kg_observed(t,i,1)*Type(1450.0)*(Type(0.0)*N_ad(t,i,0)+Type(1.0)*N_ad(t,i,1));
      lnN_egg(t,i) = log(N_egg(t,i));
    }}

  /////////////////////////////////
  //////// Process model //////////
  /////////////////////////////////


  // Fixed effects (means)
  PARAMETER(lnP_egg_riv_mean);        //Type lnP_egg_riv_mean   = -exp(-clogP_egg_riv_mean); //to avoid probabilities above 1
  PARAMETER(lnP_riv_riv_mean);        //Type P_riv_riv_mean   = exp(logitP_riv_riv_mean)/(1+exp(logitP_riv_riv_mean));
  PARAMETER(lnP_riv_ad1_mean);        //Type P_sea1_ad_mean   = exp(logitP_sea1_ad_mean)/(1+exp(logitP_sea1_ad_mean));
  PARAMETER(lnP_riv_ad2_mean);        //Type P_sea1_sea2_mean = exp(logitP_sea1_sea2_mean)/(1+exp(logitP_sea1_sea2_mean));

  // Fixed effects (variances)
    // residual
    PARAMETER(lnSD_lnN_add_residual);
    // time
    PARAMETER(lnSD_lnP_riv_sea1_t);
    PARAMETER(lnSD_lnP_sea1_sea2_t);
    // pop
    PARAMETER(lnSD_lnP_riv_ad1_pop);
    PARAMETER(lnSD_lnP_riv_ad2_pop);

  // Random effects
    // time
    PARAMETER_VECTOR(lnP_riv_sea1_t);
    PARAMETER_VECTOR(lnP_sea1_sea2_t);
    // pop
    PARAMETER_VECTOR(lnP_riv_ad1_pop);
    PARAMETER_VECTOR(lnP_riv_ad2_pop);


  // Starting values
  PARAMETER_VECTOR(lnN_riv_start);
  matrix<Type> lnN_riv(n_t, n_pop);
  for(i=0;i<n_pop;i++){
    lnN_riv(0,i) = lnN_riv_start(i);
  }

  // Defining variables
  matrix<Type> lnN_egg_surviving(n_t-1, n_pop);
  matrix<Type> lnN_staying_in_river(n_t-1, n_pop);
  matrix<Type> lnN_sea1_surviving(n_t-1, n_pop);
  matrix<Type> lnN_sea2_surviving(n_t-2, n_pop);

  matrix<Type> lnP_egg_riv(n_t-1, n_pop);
  matrix<Type> lnP_riv_riv(n_t-1, n_pop);
  matrix<Type> lnP_riv_sea1(n_t-1, n_pop);
  matrix<Type> lnP_riv_ad1(n_t-1, n_pop);
  matrix<Type> lnP_riv_ad2(n_t-2, n_pop);



  ///// Priors ////
  jnll -= dnorm(lnP_egg_riv_mean, Type(-6.0), Type(0.5), true);
  //jnll -= dnorm(lnP_riv_riv_mean, Type(-0.2), Type(0.5), true); //this must be constrained to be less than zero. Either use cloglog or hard constraints

  Type expectation=0;

  //// Model ///

  // River and Sea 1
  for(t=1;t<n_t;t++){
    for(i=0;i<n_pop;i++){

      // log Probability section
      lnP_egg_riv(t-1, i)  = lnP_egg_riv_mean;  // * P_eggTriv_t(t-1) * P_eggTriv_pop(i)
      lnP_riv_riv(t-1, i)  = lnP_riv_riv_mean;  // * P_rivTriv_t(t-1) * P_rivTriv_pop(i);
      lnP_riv_ad1(t-1, i)   = lnP_riv_ad1_mean + lnP_riv_ad1_pop(i) + lnP_riv_sea1_t(t-1);
      if(t>1 && life_history(i) != 1){
        lnP_riv_ad2(t-2, i) = lnP_riv_ad2_mean + lnP_riv_ad2_pop(i) + lnP_riv_sea1_t(t-2) + lnP_sea1_sea2_t(t-2);
      }

      // River
      lnN_egg_surviving(t-1, i) = lnN_egg(t-1,i) + lnP_egg_riv(t-1,i);
      lnN_staying_in_river(t-1, i) = lnN_riv(t-1,i) + lnP_riv_riv(t-1, i);
      lnN_riv(t,i) = log(exp(lnN_egg_surviving(t-1, i)) + exp(lnN_staying_in_river(t-1, i)));

      // Adults returning sea age 1
      expectation = lnN_riv(t-1,i) + lnP_riv_ad1(t-1,i);
      jnll -= dnorm(lnN_ad(t,i,0), expectation, exp(lnSD_lnN_add_residual), true);

      // Adults returning sea age 2
      if(t>1 && life_history(i) != 1){
      expectation = lnN_riv(t-2,i) + lnP_riv_ad2(t-2,i);
      jnll -= dnorm(lnN_ad(t,i,1), expectation, exp(lnSD_lnN_add_residual), true);
      }

    }}


  for(t=1;t<n_t;t++){
    jnll -= dnorm(lnP_riv_sea1_t(t-1), Type(0.0), exp(lnSD_lnP_riv_sea1_t), true);
    if(t>1) jnll -= dnorm(lnP_sea1_sea2_t(t-2), Type(0.0), exp(lnSD_lnP_sea1_sea2_t), true);
  }


  // stable pop diff
  for(i=0;i<n_pop;i++){
    jnll -= dnorm(lnP_riv_ad1_pop(i), Type(0.0), exp(lnSD_lnP_riv_ad1_pop), true);
    if(life_history(i) != 1){
      jnll -= dnorm(lnP_riv_ad2_pop(i), Type(0.0), exp(lnSD_lnP_riv_ad2_pop), true);
    }
  }






  return jnll;
}



