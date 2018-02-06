\name{longfused}
\alias{longfused}
\title{
Multinomial Logit Model with Lasso and Fused Lasso Penalties
}
\description{
\code{longfused} estimates the coefficients of a multinomial logit model with lasso and fused lasso penalties, using a Generalized Gradient Descent algorithm.
}
\usage{
longfused(X,Y,betaInit,lambda1,lambda2,niter,stop,eps,tauStart,factor, scaleLoss=FALSE)
}
\arguments{
  \item{X}{
	an \code{n} by \code{p} by \code{T} array of predictors, where \code{n} is the number of individuals and \code{p} is the number of predictors observed at \code{T} different times.
}
  \item{Y}{
  	an \code{n} by \code{T} matrix of outcomes, whose possible values are \code{1, \dots, K}, where \code{K} is the number of categories.
}
  \item{betaInit}{
  	a \code{(p+1)} by \code{T} by \code{(K-1)} array of initial coefficient guesses. The matrix \code{betaInit[1,,]} accomodates the initial guesses for the intercepts.
}
  \item{lambda1}{
	number: the lasso penalty.
}
  \item{lambda2}{
	number: the fused lasso penalty.
}
  \item{niter}{
  	maximum number of iterations in the Generalized Gradient Descent Algorithm.
}
  \item{stop}{
	if 1: stopping criterion on the objective function is used.
	if 2: stopping criterion on the betas is used.
}
  \item{eps}{
  	tolerance level for the stopping criterion.
}
  \item{tauStart}{
	initial step size before backtracking.
}
  \item{factor}{
	backtracking shrinkage parameter.
}
  \item{scaleLoss}{
	if TRUE the loss function is computed only on observed individuals and each term of the loss is divided by the number of completely observed individuals at time t. This fix makes the loss (roughly) independent of the effective sample size.
}


}
\details{
See the reference.
}
\value{
Returns an object of class "longfused", a list with the folowing elements:
  \item{betaHat}{
	a \code{(p+1)} by \code{T} by \code{(K-1)} array of estimated coefficients.
}
  \item{iter}{
	number of iterations of the GGD algorithm.
}
  \item{objective}{
	values of the objective function to be minimized, for each step of the GGD algorithm.
}
  \item{tau}{
	step size for each step of the GGD algorithm.
}
  \item{tau}{
	criterion's value for the stopping criterion, for each step of the GGD algorithm.
}
}
\references{
Adhikari S., Lecci F., Junker B., Tibshirani R.J., "High-dimensional Longitudinal Classification with the Multinomial Fused Lasso". Technical Report.
}
\author{
Fabrizio Lecci, Samrachana Adhikari, Ryan Tibshirani
}
\examples{
print(1)
}
