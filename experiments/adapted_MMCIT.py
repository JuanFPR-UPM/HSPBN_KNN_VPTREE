from pybnesian import IndependenceTest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
ro.r('''require("MXM")''')
ro.r('''
ci.mm_patched <- function(ind1, ind2, cs = NULL, dat, type, rob = FALSE, R = 1) {

  y <- dat[, ind1]
  x <- dat[, ind2]

  if ( is.null(cs) ) {
    ds <- data.frame(y = y, x = x)
    if ( length( unique(y) ) == 2 ) {  
      mod1 <- glm(y ~., data = ds, binomial) ## logistic regression
      t1 <- mod1$null.deviance - mod1$deviance
      dof1 <- length(mod1$coefficients) - 1
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.numeric(y) ) {
      mod1 <- lm(y ~., data = ds )  ## linear regression
      a1 <- anova(mod1)
      t1 <- a1[1, 4]
      dof1 <- a1[1, 1]
      p1 <- pf(t1, dof1, a1[2, 1], lower.tail = FALSE, log.p = TRUE)
    } else if ( is.ordered(y) ) {
      mod1 <- MXM::ordinal.reg(y ~., data = ds)  ## ordinal regression
      mod0 <- MXM::ordinal.reg(y ~ 1, data = ds)
      t1 <-  mod0$devi - mod1$devi
      dof1 <- abs( length( mod1$be ) - length( mod0$be ) )
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.factor(y)  &  !is.ordered(y) ) {  ## multinomial regression
      mod1 <- nnet::multinom(y ~., data = ds, trace = FALSE)
      mod0 <- nnet::multinom(y ~ 1, data = ds, trace = FALSE)
      a1 <- anova(mod1, mod0)
      t1 <- a1[2, 6]
      dof1 <- a1[2, 5]
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)
    } 
    ###### other direction now
    if ( length( unique(x) ) == 2) {
      mod2 <- glm(x ~., data = ds, binomial)  ## logistic regression
      t2 <- mod2$null.deviance - mod2$deviance
      dof2 <- length(mod2$coefficients) - 1
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.numeric(x) ) {
      mod2 <- lm(x ~., data = ds ) ## linear regression
      a2 <- anova(mod2)
      t2 <- a2[1, 4]
      dof2 <- a2[1, 1]
      p2 <- pf(t2, dof2, a2[2, 1], lower.tail = FALSE, log.p = TRUE)
    } else if ( is.ordered(x) ) {
      mod2 <- MXM::ordinal.reg(x ~., data = ds)  ## ordinal regression
      mod0 <- MXM::ordinal.reg(x ~ 1, data = ds)
      t2 <- mod0$devi - mod2$devi
      dof2 <- abs( length( mod2$be ) - length( mod0$be ) )
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.factor(x)  &  !is.ordered(x) ) {
      mod2 <- nnet::multinom(x ~., data = ds, trace = FALSE)  ## multinomial regression
      mod0 <- nnet::multinom(x ~ 1, data = ds, trace = FALSE)
      a2 <- anova(mod2, mod0)
      t2 <- a2[2, 6]
      dof2 <- a2[2, 5]
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)
    } 
  } else {
    ### with conditioning set   
    ds0 <- dat[, cs, drop = FALSE ]
    ds1 <- dat[, c(cs, ind2) ]
    if ( length( unique(y) ) == 2 ) { 
      mod1 <- glm(y ~., data = ds1, binomial)  ## logistic regression
      mod0 <- glm(y ~., data = ds0, binomial)
      a1 <- anova(mod0, mod1)
      t1 <- a1[2, 4]
      dof1 <- a1[2, 3]
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)

  # Replace the is.numeric(y) block with:
  } else if ( is.numeric(y) ) {
      mod1 <- lm(y ~., data = ds1)
      mod0 <- lm(y ~., data = ds0)
      a1 <- anova(mod0, mod1)
      if (is.na(a1[2, 5])) { 
          p1 <- log(1) # x was perfectly collinear and dropped
      } else {
          # Calculate log p-value from the F-statistic (column 5)
          p1 <- pf(a1[2, 5], a1[2, 3], a1[2, 1], lower.tail = FALSE, log.p = TRUE)
      } 
    } else if ( is.ordered(y) ) {
      mod1 <- MXM::ordinal.reg(y ~., data = ds1)   ## ordinal regression
      mod0 <- MXM::ordinal.reg(y ~., data = ds0)
      t1 <- mod0$devi - mod1$devi
      if (t1 <0 )  t1 <- 0
      dof1 <- abs( length( mod1$be ) - length( mod0$be ) )
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.factor(y)  &  !is.ordered(y) ) {
      mod1 <- nnet::multinom(y ~., data = ds1, trace = FALSE)  ## multinomial regression
      mod0 <- nnet::multinom(y ~., data = ds0, trace = FALSE)
      t1 <- deviance(mod0) - deviance(mod1)
      dof1 <- length( coef(mod1) ) - length( coef(mod0) )
      p1 <- pchisq(t1, dof1, lower.tail = FALSE, log.p = TRUE)
    } 
    ###### other direction now
    ds2 <- dat[, c(cs, ind1) ]
    if ( length( unique(x) ) == 2 ) {
      mod2 <- glm(x ~., data = ds2, binomial)  ## binomial regression
      mod0 <- glm(x ~., data = ds0, binomial)
      a2 <- anova(mod0, mod2)
      t2 <- a2[2, 4]
      dof2 <- a2[2, 3]
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)

  # Replace the is.numeric(x) block with:
  } else if ( is.numeric(x) ) {
      mod2 <- lm(x ~., data = ds2)
      mod0 <- lm(x ~., data = ds0)
      a2 <- anova(mod0, mod2)
      if (is.na(a2[2, 5])) {
          p2 <- log(1)
      } else {
          p2 <- pf(a2[2, 5], a2[2, 3], a2[2, 1], lower.tail = FALSE, log.p = TRUE)
      }
  }
    else if ( is.ordered(x) ) {
      mod2 <- MXM::ordinal.reg(x ~., data = ds2)  ## ordinal regression
      mod0 <- MXM::ordinal.reg(x ~., data = ds0)
      t2 <- mod0$devi - mod2$devi
      if (t2 <0 )  t2 <- 0
      dof2 <- abs( length( mod2$be ) - length( mod0$be ) )
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)
    } else if ( is.factor(x)  &  !is.ordered(x) ) {
      mod2 <- nnet::multinom(x ~., data = ds2, trace = FALSE)  ## multinomial regression
      mod0 <- nnet::multinom(x ~., data = ds0, trace = FALSE)
      t2 <- deviance(mod0) - deviance(mod2)
      dof2 <- length( coef(mod2) ) - length( coef(mod0) )
      p2 <- pchisq(t2, dof2, lower.tail = FALSE, log.p = TRUE)
    } 
  }   
  pval <- min( log(2) + min(p1, p2), max(p1, p2) )
  stat <-  max( 2 * max(t1, t2), min(t1, t2) )
  if (p1 < p2) { 
    dof <- dof1
  } else  dof <- dof2
  result <- c(stat, pval, dof)
  names(result) <- c('test', 'logged.p-value', 'df') 
  result
}

# Overwrite the MXM namespace to allow correlated numerical features
assignInNamespace("ci.mm", ci.mm_patched, ns="MXM")
''')

class MMCIT(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df

        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
        ro.globalenv['df_r'] = df_r

        # Assuming ci.mm is available in in R environment inside the MXM package

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        x = np.where(self.df.columns == x)[0][0]
        y = np.where(self.df.columns == y)[0][0]

        if z is not None and len(z) > 0:
            if isinstance(z, str):
                z = [z]
            z_str = "c(" + ",".join([str(np.where(self.df.columns ==
                                                  col)[0][0] + 1) for col in z]) + ")"
        else:
            z_str = "NULL"

        r_call = f"ci.mm({int(x)+1}, {int(y)+1}, cs={z_str}, dat=df_r)"

        res = ro.r(r_call)
        log_pval = res.rx2('logged.p-value')[0]
        return np.exp(log_pval)    # exponentiate logvalue
