\name{plot.longfused}
\alias{plot.longfused}
\title{
  Plot longfused estimated coefficients.
}
\description{
This function plots the estimated coefficients stored in an object of class \code{longfused}. 
}
\usage{
\method{plot}{longfused}(x, intercept=FALSE, levels=NULL, beta=NULL, ...)
}
\arguments{
  \item{x}{
	an object of class "longfused", as returned by the function \code{\link{longfused}}.
}
  \item{intercept}{
	if \code{TRUE} the intercept is displayed.
}
  \item{levels}{
  	a vector of strings for the levels of the outcome variable in the longfused model.
}
  \item{beta}{
 Optional: a \code{(p+1)} by \code{T} by \code{(K-1)} array of true coefficients to be displayed along with the estimated coefficients. Default is \code{NULL} and the true coefficients are not displayed.
}
  \item{\dots}{
    additional graphical parameters.
}
}
\author{
Fabrizio Lecci
}
\examples{
print(1)
}
