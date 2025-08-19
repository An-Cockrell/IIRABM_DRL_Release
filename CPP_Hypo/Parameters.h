	const int xDim=101;
	const int yDim=101;

	const int numTimeSteps=10000; //5760=28 days

	const int injuryStep=205;
	const int cellCapacity=28;
	const int numRules=25;
	const int numRuleParams=17;

//PK/PD variables
const float etanerceptReduction=0.9;
const float rilanoceptReduction=0.9;
const float dupilumabReduction=0.9;
const float ustekinumabReduction=0.9;

const float etanerceptElimConstant=0.0069;
const float rilanoceptElimConstant=0.0041;
const float filgrastimElimConstant=0.1980;
const float actimmuneElimConstant=1.0502;
const float dupilumabElimConstant=0.0021;
const float ustekinumabElimConstant=0.0011;
