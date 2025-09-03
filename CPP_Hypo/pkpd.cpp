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
extern const int cellCapacity,xDim,yDim,injuryStep,parameterInput,numTimeSteps,step;
extern const float etanerceptReduction,rilanoceptReduction,dupilumabReduction,ustekinumabReduction;
extern const float etanerceptElimConstant,rilanoceptElimConstant,filgrastimElimConstant,
                   actimmuneElimConstant,dupilumabElimConstant,ustekinumabElimConstant;

int drugTimings[6]; //etanercept,rilonacept, filgrastim, actimmune, dupilumab, ustekinumab;
float drugConcentrations[6];

//Etanercept - binds to and neutralizes TNF >90% for enture drug course; Half-life =100 h^{-1}; elim constant=0.0069
//1) https://pmc.ncbi.nlm.nih.gov/articles/PMC3726066/
//2) Update on the use of etanercept across a spectrum of rheumatoid disorders
//3) Effects of three anti-TNF-α drugs: Etanercept, infliximab and pirfenidone on release of TNF-α in medium and TNF-α associated with the cell in vitro
//   https://www.sciencedirect.com/science/article/abs/pii/S1567576908000155

//rilonacept - neutralizes IL_1; half-life=168 h^{-1}; elim constant=0.0041

//filgrastim (Neupogen) - same as giving GCSF; increases GCSF by ~1000x, half-life = 3.5 h^{-1}; elim constant=0.1980

//Actimmune - same as giving IFNg; half-life = 0.66 h^{-1}; elim constant=1.0502

//Dupilumab (Dupixent) - inhibits IL4 signaling (does not reduce IL4) - half-life=335 h^{-1}; elim constant=0.0021

//Ustekinumab (Stelara) - reduces IL12 levels - half-life = 648 h^{-1}; elim constant=0.0011
//P663 Post-ustekinumab induction IL12, IL23, and ustekinumab levels are associated with clinical response in a multi-centre prospective cohort study of Crohn’s disease patients: results from the AURORA Study including ANZIBDC Cohort
//https://academic.oup.com/ecco-jcc/article/17/Supplement_1/i793/7010121



float oneCompartmentPKPD(float volume, float initialDose, float timeElapsed, float eliminationConstant){
  //Mortensen, S.B., Jónsdóttir, A.H., Klim, S. and Madsen, H., 2008. Introduction to PK/PD modelling-with focus on PK and stochastic differential equations.
  float answer;
  answer=initialDose/volume*exp(-eliminationConstant*timeElapsed);
  return answer;
}

void updateDrugConcentrations(){
  float timeElapsed;
  if(drugConcentrations[0]>0){
    timeElapsed=(step-drugTimings[0])/10;
    drugConcentrations[0]=oneCompartmentPKPD(1,1,timeElapsed,etanerceptElimConstant)
  }

  if(drugConcentrations[1]>0){
    timeElapsed=(step-drugTimings[1])/10;
    drugConcentrations[1]=oneCompartmentPKPD(1,1,timeElapsed,rilanoceptElimConstant
  }

  // if(drugConcentrations[2]>0){
  //   timeElapsed=(step-drugTimings[2])/10;
  //   drugConcentrations[2]=oneCompartmentPKPD(1,1,timeElapsed,filgrastimElimConstant)
  // }
  //
  // if(drugConcentrations[3]>0){
  //   timeElapsed=(step-drugTimings[3])/10;
  //   drugConcentrations[3]=oneCompartmentPKPD(1,1,timeElapsed,actimmuneElimConstant)
  // }

  if(drugConcentrations[4]>0){
    timeElapsed=(step-drugTimings[4])/10;
    drugConcentrations[4]=oneCompartmentPKPD(1,1,timeElapsed,dupilumabElimConstant)
  }

  if(drugConcentrations[5]>0){
    timeElapsed=(step-drugTimings[5])/10;
    drugConcentrations[5]=oneCompartmentPKPD(1,1,timeElapsed,ustekinumabElimConstant)
  }
}

void etanerceptEffect(){
  float ER;
  ER=etanerceptReduction;
  if(drugConcentrations[0]<0.33){
    ER=etanerceptReduction*(3*drugConcentration[0]);
  }
  for(i=0;i<totalCells;i++){
    ecArray[i].TNF-=(ER*ecArray[i].TNF);
  }
}

void rilonaceptEffect(){
  float RR;
  RR=rilonaceptReduction;
  if(drugConcentrations[0]<0.5){
    RR=rilonaceptReduction*(2*drugConcentration[1]);
  }
  for(i=0;i<totalCells;i++){
    ecArray[i].IL1-=(RR*ecArray[i].IL1);
  }
}

void dipulmabEffect(){
  float DR;
  DR=dipulmabReduction;
  if(drugConcentrations[0]<0.2){
    DR=dipulmabReduction*(5*drugConcentration[4]);
  }
  for(i=0;i<totalCells;i++){
    ecArray[i].IL4-=(DR*ecArray[i].IL4);
  }
}

void ustekinumabEffect(){
  for(i=0;i<totalCells;i++){
    ecArray[i].IL12-=(ustekinumabReduction*ecArray[i].IL12);
  }
}

void applyEtanercept(){
  drugConcentrations[0]=1.0;
  drugTimings[0]=step;
}

void applyRilonacept(){
  drugConcentrations[1]=1.0;
  drugTimings[1]=step;
}

// void applyFilgrastim(){
//   // drugConcentrations[2]=1.0;
//   // drugTimings[2]=step;
//   for(i=0;i<totalCells;i++){
//     ecArray[i].GCSF+=10;
//   }
// }
//
// void applyActimmune(){
//   // drugConcentrations[3]=1.0;
//   // drugTimings[3]=step;
//   for(i=0;i<totalCells;i++){
//     ecArray[i].IFNg+=5;
//   }
// }

void applyDupilumab(){
  drugConcentrations[4]=1.0;
  drugTimings[4]=step;
}

void applyUstekinumab(){
  drugConcentrations[5]=1.0;
  drugTimings[5]=step;
}
