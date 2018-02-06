// Fabrizio Lecci, lecci at cmu dot edu
// 
// GGD.cpp
//
// Generalized Gradient Descent Algorithm for Fused lasso with PP predictors
// at TT different ages and Multinomial logistic likelihood, with KK+1 levels.
// The intercept is stored in XX, so the penalties start from j=2
// See paper for more details


#include <R.h>
#include <R_ext/Print.h>
#include <iostream>
#include <cmath>


using namespace std;

//Declaration
double vecMlt(double *XX, double *YY, int NN);
double norm2(double *XX, int NN);
double array3MltLineCol(double *XX, double *Beta, int *pNN, int *pPP, int *pTT, int *pKK, int ii, int tt, int hh);

void obj_h(int *pPP, int *pTT, int *pKK, double *beta, double *plambda1, double *plambda2, double* pHH);
void obj_g(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *pGG);
void obj_fnc(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *plambda1, double *plambda2, double *result);
void delta(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *resDelta);

void DP_for_1DFL(int n, double *y, double lam, double *beta);
void soft_thresh(int n, double *y, double lam, double *beta);
void DP_for_S1DFL(int n, double *y, double lam1, double lam2, double *beta);



// read element of matrix
double ReadMat(double*XX, int *pNN, int *pDD, int i, int d){
	double out=0;
	out=XX[(d-1)*(pNN[0])+i-1];
	return out;
}

// write element of matrix
void WriteMat(double*XX, int *pNN, int *pDD, int i, int d, double input){
	XX[(d-1)*(pNN[0])+i-1]=input;
}

// read element of array
double ReadArr(double*XX, int *pNN, int *pDD, int *pKK, int ii, int jj, int kk){
	double out=0;
	out=XX[(kk-1)*(pNN[0])*(pDD[0])+(jj-1)*(pNN[0])+ii-1];
	return out;
}

// write element of array
void WriteArr(double*XX, int *pNN, int *pDD, int *pKK, int ii, int jj, int kk, double input){
	XX[(kk-1)*(pNN[0])*(pDD[0])+(jj-1)*(pNN[0])+ii-1]=input;
}


// vecMlt returns the product of two vectors of length NN
double vecMlt(double *XX, double *YY, int NN){
    double res=0;
    
    for (int i=1; i<=NN; i++) {
        res += XX[i-1]*YY[i-1];
    }
    
    return res;
}


//norm2 returns the squared norm of a vector
double norm2(double *XX, int NN){
    double res=0;
    
    for (int i=1; i<=NN; i++) {
        res += XX[i-1]*XX[i-1];
    }
    
    return res;
}


//Given a NN*PP*TT array XX and a PP*TT*KK array Beta, array3MltLineCol returns the product XX[ii,,tt] *  Beta[,tt,hh]
double array3MltLineCol(double *XX, double *Beta, int *pNN, int *pPP, int *pTT, int *pKK, int ii, int tt, int hh){
    double res=0;
    
    for (int j=1; j<=pPP[0]; j++) {
        res +=  ReadArr(XX, pNN, pPP, pTT, ii, j, tt)*
        ReadArr(Beta, pPP, pTT, pKK, j, tt, hh);
    }
    
    return res;
}


//delta returns the gradient of the likelihood wrt the coefficients
void delta(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *resDelta){
    
    double firstTerm=0.0;
    double secondTerm=0.0;
    double twoTerms=0.0;
    
    // We store the gradient into an array of dimension PP*TT*KK
    for (int k=1; k<=pKK[0]; k++) {
        for (int t=1; t<=pTT[0]; t++) {
            for (int j=1; j<=pPP[0]; j++) {
                
                twoTerms=0.0;
                for (int i=1; i<=pNN[0]; i++) {
                    if (ReadMat(YY, pNN, pTT, i, t)!=0){  //check that individual i has diagnosis at time t
                        firstTerm=0.;
                        secondTerm=0.;
                        
                        firstTerm =  -(ReadMat(YY, pNN, pTT, i, t)==k )* ReadArr(XX, pNN, pPP, pTT, i, j, t);
                        
                        for (int h=1; h<=pKK[0]; h++) {
                            secondTerm += exp(array3MltLineCol(XX, beta, pNN, pPP, pTT, pKK, i, t, h));
                        }
                        
                        
                        secondTerm= ReadArr(XX, pNN, pPP, pTT, i, j, t) *  exp(array3MltLineCol(XX, beta, pNN, pPP, pTT, pKK, i, t, k)) / (1+secondTerm);
                        twoTerms+= firstTerm+ secondTerm;
                    }
                }
                
                WriteArr(resDelta, pPP, pTT, pKK, j, t, k, twoTerms/Nt[t-1]);
            }
        }
    }
    
    return ;
}


// newBeta returns the updated matrix of coefficients for each step of the GGD algorithm
void newBeta(double *beta, double* DELTA, double *tau, int *pNN, int *pPP, int *pTT, int *pKK, double *lambda1, double *lambda2, double *resBeta){
    
    double *yy= new double[pTT[0]];                          //pass this to the function 
    double *tBeta= new double[pTT[0]];
    
    // for each j (starting from 2) and each k we update Beta, by 1d fused lasso
    for (int k=1; k<=(pKK[0]); k++) {
        for (int j=2; j<=pPP[0]; j++) {
        
            for (int t=1; t<=(pTT[0]); t++) {
                yy[t-1]=ReadArr(beta, pPP, pTT, pKK, j, t, k)
		                - tau[0]* ReadArr(DELTA, pPP, pTT, pKK, j, t, k);
              
            }
            DP_for_S1DFL(pTT[0], yy, lambda1[0], lambda2[0], tBeta);
            
            for (int t=1; t<=(pTT[0]); t++) {
                WriteArr(resBeta, pPP, pTT, pKK, j, t, k, tBeta[t-1]);
            }
        }
    }
    
    // for j=1, each beta is manually updated, because we do not penalize the intercept
    for (int t=1; t<=(pTT[0]); t++) {
        for (int k=1; k<=(pKK[0]); k++) {
        WriteArr(resBeta, pPP, pTT, pKK, 1, t, k, ReadArr(beta, pPP, pTT, pKK, 1, t, k)- tau[0] * ReadArr(DELTA, pPP, pTT, pKK, 1, t, k));
        }
    }
    
    delete[] yy;
    delete[] tBeta;
}


// obj_fnc returns the value of -likelihood+penalty
void obj_fnc(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *plambda1, double *plambda2, double *result){
    double GG;
    double HH;
    double* pGG;
    double* pHH;
    pGG=&GG;
    pHH=&HH;
    
    obj_h(pPP, pTT, pKK, beta, plambda1, plambda2, pHH);
    obj_g(YY, XX, beta, pNN, pPP, pTT, pKK, Nt, pGG);
    result[0]=pGG[0]+pHH[0];
    return;
}


// obj_h returns the value of the penalty
void obj_h(int *pPP, int *pTT, int *pKK, double *beta, double *plambda1, double *plambda2, double* pHH){
    pHH[0]=0.0;

	//loop for lasso
    for (int k=1; k<=pKK[0]; k++) {
        for (int t=1; t<=pTT[0]; t++) {
            for (int j=2; j<=pPP[0]; j++) {
                pHH[0]+= plambda1[0]* abs(ReadArr(beta, pPP, pTT, pKK, j, t, k));
            }
        }
    }
	//loop for fused lasso
    for (int k=1; k<=pKK[0]; k++) {
        for (int t=1; t<=(pTT[0]-1); t++) {
            for (int j=2; j<=pPP[0]; j++) {
                pHH[0]+= 
                plambda2[0] * abs(ReadArr(beta, pPP, pTT, pKK, j, (t+1), k)- ReadArr(beta, pPP, pTT, pKK, j, t, k));
            }
        }
    }

    return ;
}


// obj_g returns the value of -likelihood
void obj_g(double *YY, double *XX, double *beta, int *pNN, int *pPP, int *pTT, int *pKK, int *Nt, double *pGG){
    double firstTerm=0.0;
    double logTerm=0.0;
    double twoTerms=0.0;
    pGG[0]=0.0;
    
    for (int t=1; t<=pTT[0]; t++) {
        twoTerms=0.0;
        for (int i=1; i<=pNN[0]; i++) {
            if (ReadMat(YY, pNN, pTT, i, t)!=0){  //check that individual i has diagnosis at time t
                firstTerm=0.;
                logTerm=0;
                
                for (int k=1; k<=pKK[0]; k++) {
                    firstTerm +=  -(ReadMat(YY, pNN, pTT, i, t)==k )*
                    array3MltLineCol(XX, beta, pNN, pPP, pTT, pKK, i, t, k);
                }
                
                for (int h=1; h<=pKK[0]; h++) {
                    logTerm += exp(array3MltLineCol(XX, beta, pNN, pPP, pTT, pKK, i, t, h));
                    
                }
                
                logTerm=(log(1+logTerm));
                twoTerms+= firstTerm+logTerm;
            }
        }
        pGG[0] += twoTerms/Nt[t-1];
    }
    return ;
}


// backtracking returns the optimal tau for the GGD algorithm
// It also returns the updated Beta associated with the optimal tau
void backtracking(double *YY, double *XX, double* pTau, double* pFactor, double* beta, double* DELTA, double* lambda1, double* lambda2, int *pNN, int *pPP, int *pTT, int *pKK, int *pNt, double* resTau, double* new_Beta ){
    
    double lam1;
    double lam2;
    double* pLam1= &lam1;
    double* pLam2= &lam2;
    double F1=1;
    double* pF1=&F1;
    double F2=0;
    double* pF2=&F2;
    double firstF2=0;
    double* pFirstF2=&firstF2;

    
    double* GG=0;
    GG= new double[pPP[0]*pTT[0]*pKK[0]];
    
    // initial values
    resTau[0]= pTau[0];
    lam1=resTau[0]*lambda1[0];
    lam2=resTau[0]*lambda2[0];
    
    
    //backtracking algorithm
    int b = 1;
    while(pF1[0]>pF2[0]&& b < 50){
        b = b + 1;
        //update beta
        newBeta(beta, DELTA, resTau, pNN, pPP, pTT, pKK, pLam1, pLam2, new_Beta);
        
        for (int i=1; i<=(pPP[0]*pTT[0]*pKK[0]); i++) {
            GG[i-1]=(beta[i-1]-new_Beta[i-1])/resTau[0];
        }
        
        // compute F1
        obj_g(YY, XX, new_Beta, pNN, pPP, pTT, pKK, pNt, pF1);
        
        //compute F2
        obj_g(YY, XX, beta, pNN, pPP, pTT, pKK, pNt, pFirstF2);
        F2=pFirstF2[0]-
        resTau[0]*vecMlt(DELTA, GG, pPP[0]*pTT[0]*pKK[0])+
        (resTau[0]/2)* norm2(GG, pPP[0]*pTT[0]*pKK[0]) ;
        
        //update tau for the next step, in the case that F1>F2
        resTau[0]=pFactor[0]*resTau[0];
        lam1=resTau[0]*lambda1[0];
        lam2=resTau[0]*lambda2[0];
      
    }
    
    // the optimal tau is the penultimate one
    resTau[0]=resTau[0]/pFactor[0];
    
    delete[] GG;
    
}
    



// Ryan Tibshirani's 1d fused lasso

// Dynamic programming algorithm for the 1d fused lasso problem
// (Idea by Nick Johnson)
void DP_for_1DFL(int n, double *y, double lam, double *beta) {
    // Take care of a few trivial cases
    if (n==0) return;
    if (n==1 || lam==0) {
        for (int i=0; i<n; i++) beta[i] = y[i];
        return;
    }
    
    // These are used to store the derivative of the
    // piecewise quadratic function of interest
    double afirst, alast, bfirst, blast;
    double *x = (double*)malloc(2*n*sizeof(double));
    double *a = (double*)malloc(2*n*sizeof(double));
    double *b = (double*)malloc(2*n*sizeof(double));
    int l,r;
    
    // These are the knots of the back-pointers
    double *tm = (double*)malloc((n-1)*sizeof(double));
    double *tp = (double*)malloc((n-1)*sizeof(double));
    
    // We step through the first iteration manually
    tm[0] = -lam+y[0];
    tp[0] = lam+y[0];
    l = n-1;
    r = n;
    x[l] = tm[0];
    x[r] = tp[0];
    a[l] = 1;
    b[l] = -y[0]+lam;
    a[r] = -1;
    b[r] = y[0]+lam;
    afirst = 1;
    bfirst = -lam-y[1];
    alast = -1;
    blast = -lam+y[1];
    
    // Now iterations 2 through n-1
    int lo, hi;
    double alo, blo, ahi, bhi;
    for (int k=1; k<n-1; k++) {
        // Compute lo: step up from l until the
        // derivative is greater than -lam
        alo = afirst;
        blo = bfirst;
        for (lo=l; lo<=r; lo++) {
            if (alo*x[lo]+blo > -lam) break;
            alo += a[lo];
            blo += b[lo];
        }
        
        // Compute the negative knot
        tm[k] = (-lam-blo)/alo;
        l = lo-1;
        x[l] = tm[k];
        
        // Compute hi: step down from r until the
        // derivative is less than lam
        ahi = alast;
        bhi = blast;
        for (hi=r; hi>=l; hi--) {
            if (-ahi*x[hi]-bhi < lam) break;
            ahi += a[hi];
            bhi += b[hi];
        }
        
        // Compute the positive knot
        tp[k] = (lam+bhi)/(-ahi);
        r = hi+1;
        x[r] = tp[k];
        
        // Update a and b
        a[l] = alo;
        b[l] = blo+lam;
        a[r] = ahi;
        b[r] = bhi+lam;
        afirst = 1;
        bfirst = -lam-y[k+1];
        alast = -1;
        blast = -lam+y[k+1];
    }
    
    // Compute the last coefficient: this is where
    // the function has zero derivative
    
    alo = afirst;
    blo = bfirst;
    for (lo=l; lo<=r; lo++) {
        if (alo*x[lo]+blo > 0) break;
        alo += a[lo];
        blo += b[lo];
    }
    beta[n-1] = -blo/alo;
    
    // Compute the rest of the coefficients, by the
    // back-pointers
    for (int k=n-2; k>=0; k--) {
        if (beta[k+1]>tp[k]) beta[k] = tp[k];
        else if (beta[k+1]<tm[k]) beta[k] = tm[k];
        else beta[k] = beta[k+1];
    }
    
    // Done! Free up memory
    free(x);
    free(a);
    free(b);
    free(tm);
    free(tp);
}

// Soft-thresholding
void soft_thresh(int n, double *y, double lam, double *beta) {
    for (int i=0; i<n; i++) {
        if (y[i]>lam) beta[i] = y[i]-lam;
        else if (y[i]<-lam) beta[i] = y[i]+lam;
        else beta[i]=0;
    }
}

// The sparse 1d fused lasso
void DP_for_S1DFL(int n, double *y, double lam1, double lam2, double *beta) {
    DP_for_1DFL(n,y,lam2,beta);
    soft_thresh(n,beta,lam1,beta);
}

// End of Ryan Tibshirani's 1d fused lasso






// R wrapper

extern "C" {
   
    // GGD algorithm
    void GGD(double *YY, double *XX, double* beta, double* lambda1, double* lambda2, int *pNN, int *pPP, int *pTT, int *pKK, int *pNt, int* pNiter, int* pStop, double* pEps, double* pTau, double* pFactor, double* new_Beta, double* objectives, double* taus, double* criterions, int *actualNiter){

        double* DELTA=0;
        DELTA= new double[pPP[0]*pTT[0]*pKK[0]];
        double* betaCriterion=0;
        betaCriterion= new double[pPP[0]*pTT[0]*pKK[0]];
        double stopping=pEps[0]+1;
        double resTau=0.;
        double* presTau= &resTau;
        double beforeF=0.;
        double* pbeforeF= &beforeF;
        double afterF=0.;
        double* pafterF= &afterF;
        
        // step counter
        int ss=1;
        
        //store current beta in new beta
        for (int i=1; i<=pPP[0]*pTT[0]*pKK[0]; i++) {
            new_Beta[i-1]=beta[i-1];
        }
        
        while (ss <= pNiter[0] && stopping>pEps[0]) {
            
            // the new beta becomes the old beta for the new step
            for (int i=1; i<=pPP[0]*pTT[0]*pKK[0]; i++) {
                beta[i-1]=new_Beta[i-1];
            }
            
            //compute the gradient
            delta(YY, XX, beta, pNN, pPP, pTT, pKK, pNt, DELTA);
            
            //value of the objective function before updating beta
            if (ss==1) {
                obj_fnc(YY, XX, beta, pNN, pPP, pTT, pKK, pNt, lambda1, lambda2, pbeforeF);
            }
            else{
                pbeforeF[0]=pafterF[0];
            }
            
            // backtracking algorithm returns optimal tau and new beta
            backtracking(YY, XX, pTau, pFactor, beta, DELTA, lambda1, lambda2, pNN, pPP, pTT, pKK, pNt, presTau, new_Beta );
            
            // value of the objective function after updating beta
            obj_fnc(YY, XX, new_Beta, pNN, pPP, pTT, pKK, pNt, lambda1, lambda2, pafterF);
            
            // Stopping criterion
            if (pStop[0]==1) {
                // stopping criterion on the objective function
                stopping=abs(pafterF[0]-pbeforeF[0])/pbeforeF[0];
            }
            else if(pStop[0]==2) {
                //numerator for stopping criterion on beta
                for (int i=1; i<=pPP[0]*pTT[0]*pKK[0]; i++) {
                    betaCriterion[i-1]=new_Beta[i-1]-beta[i-1];
                }                
                stopping=(sqrt(norm2(betaCriterion, pPP[0]*pTT[0]*pKK[0])))/sqrt(norm2(beta, pPP[0]*pTT[0]*pKK[0]));
            }
            else{
                // the algorithm will run until the maximum number of iterations
                stopping=pEps[0]+1;
            }
            
            // the function returns the final estimated beta and the following values for each step
            taus[ss-1]=presTau[0];
            objectives[ss-1]=pafterF[0];
            criterions[ss-1]=stopping;
            
            //print ss and update it for next step
            Rprintf("%d ", ss);
            ss=ss+ 1;
        }
        
        Rprintf("\n");
        
        actualNiter[0]=ss-1;
        
        delete[] DELTA;
        delete[] betaCriterion;
    }

}
    


