longfused<-
function(X,Y,betaInit,lambda1,lambda2,niter,stop,eps,tauStart,factor, scaleLoss=FALSE){
	NN=dim(X)[1]
	PP=dim(X)[2]
	TT=dim(X)[3]

	## add a column to X for the intercept
	XwithIntercept=array(1,c(NN,PP+1,TT))
	XwithIntercept[,-1,]=X
	X=XwithIntercept
	PP=PP+1

	KKminus1=dim(betaInit)[3]
	
	if (scaleLoss)
	{
		#only consider non scaleLoss values
		Nt = sapply(1:TT, FUN= function(tt) sum(!is.na(Y[,tt])) )  
		Y[which(is.na(Y))]=0
	} else
	{
		Nt= rep(1, TT)	
	}
	
	newBeta=betaInit	
	objectives=rep(0,niter)
	taus=rep(0,niter)
	criterions=rep(0,niter)
	actualNiter=0
			
	out=.C("GGD", as.double(Y), as.double(X), as.double(betaInit), as.double(lambda1), as.double(lambda2), as.integer(NN), as.integer(PP), as.integer(TT), as.integer(KKminus1), as.integer(Nt), as.integer(niter), as.integer(stop), as.double(eps), as.double(tauStart), as.double(factor), as.double(newBeta), as.double(objectives), as.double(taus), as.double(criterions), as.integer(actualNiter),dup=FALSE )
	
	actualNiter=out[[20]]
	newBeta=array(out[[16]], c(PP,TT,KKminus1))
	objectives=out[[17]][1:actualNiter]	
	taus=out[[18]][1:actualNiter]	
	criterions=out[[19]][1:actualNiter]	
	
	out=list("betaHat"=newBeta, "iter"=actualNiter, "objective"=objectives, "tau"=taus, "StopCriterion"=criterions)
	class(out)="longfused"
	return(out)	
}
