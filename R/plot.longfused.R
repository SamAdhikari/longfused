plot.longfused<-
function(x, intercept=FALSE, levels=NULL, beta=NULL, ...)
{	
	p = dim(x$betaHat)[1]
	TT=dim(x$betaHat)[2]
	KK=dim(x$betaHat)[3]+1
	ylimPlot= c(min(x$betaHat[,,],beta)-0.01,max(x$betaHat[,,],beta)+0.01)
	
	## re-code the intercept
	if (intercept==FALSE) 
	{
		intercept=2
		ylimPlot= c(min(x$betaHat[-1,,],beta)-0.01,
					max(x$betaHat[-1,,],beta)+0.01)
	}
	
	par(mfrow=c(1,KK-1))
	for (k in 1:(KK-1)){
		plot((1:TT),x$betaHat[1,,k],type='n',ylim=ylimPlot, 
		xlim = c(-2,TT),xlab="Time",ylab="Estimated Beta",axes = FALSE, 
		main = paste(levels[k], " vs ", levels[KK], sep="  "), ... )
		axis(1, )
		axis(2, )

		for (i in intercept:p){
			t=1:TT
			lines(t,x$betaHat[i,,k], col = i, type = "l", lty=2,lwd=2)
		}

		if (!is.null(beta))
		{
			for (i in intercept:p)
			{
				lines(1:TT,beta[i,,k],col = i,type = "l", lty=1, lwd=1)
			}
			legend(1,ylimPlot[2], c("true Beta", "estimated Beta"), lty=c(1,2), lwd=c(1,2))	
		}
	}
}
