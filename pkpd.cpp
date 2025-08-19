#include <vector>
#include <random>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include "agents.h"
#include "Simulation.h"
#include "Parameters.h"

using namespace std;

extern vector<EC> ecArray;
extern const int cellCapacity,xDim,yDim,injuryStep,parameterInput,numTimeSteps;
extern const float etanerceptReduction,rilanoceptReduction,dupilumabReduction,ustekinumabReduction;

//Etanercept - binds to and neutralizes TNF >90% for enture drug course; Half-life =100 h^{-1}
//1) https://pmc.ncbi.nlm.nih.gov/articles/PMC3726066/
//2) Update on the use of etanercept across a spectrum of rheumatoid disorders
//3) Effects of three anti-TNF-α drugs: Etanercept, infliximab and pirfenidone on release of TNF-α in medium and TNF-α associated with the cell in vitro
//   https://www.sciencedirect.com/science/article/abs/pii/S1567576908000155

//rilonacept - neutralizes IL_1; half-life=168 h^{-1}

//filgrastim (Neupogen) - same as giving GCSF; increases GCSF by ~1000x, half-life = 3.5 h^{-1}

//Actimmune - same as giving IFNg; half-life = 0.66 h^{-1}

//Dupilumab (Dupixent) - inhibits IL4 signaling (does not reduce IL4) - half-life=335 h^{-1}

//Ustekinumab (Stelara) - reduces IL12 levels - half-life = 648 h^{-1}
//P663 Post-ustekinumab induction IL12, IL23, and ustekinumab levels are associated with clinical response in a multi-centre prospective cohort study of Crohn’s disease patients: results from the AURORA Study including ANZIBDC Cohort
//https://academic.oup.com/ecco-jcc/article/17/Supplement_1/i793/7010121



float oneCompartmentPKPD(float volume, float initialDose, float timeElapsed, float eliminationConstant){
  //Mortensen, S.B., Jónsdóttir, A.H., Klim, S. and Madsen, H., 2008. Introduction to PK/PD modelling-with focus on PK and stochastic differential equations.
  float answer;
  answer=initialDose/volume*exp(-eliminationConstant*timeElapsed);
  return answer;

}

void etanerceptEffect(){
  for(i=0;i<totalCells;i++){
    ecArray[i].TNF-=(etanerceptReduction*ecArray[i].TNF);
  }
}

void rilonaceptEffect(){
  for(i=0;i<totalCells;i++){
    ecArray[i].IL1-=(rilonaceptReduction*ecArray[i].IL1);
  }
}

void dipulmabEffect(){
  for(i=0;i<totalCells;i++){
    ecArray[i].IL4-=(dupilumabReduction*ecArray[i].IL4);
  }
}
