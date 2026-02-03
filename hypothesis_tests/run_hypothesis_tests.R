load("Var.RData")


# LONG SHORT

nl_rl_daily_rets_longshort <- read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR/NonLinear_Shrinkage/results/model_NL_daily_returns.csv",
                                     sep = ",")
nl_base_daily_rets_longshort <-  read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR/NonLinear_Shrinkage/results/QIS_base_daily_returns.csv",
                                           sep = ",")
nl_base_daily_rets_longshort <- nl_base_daily_rets_longshort[5041:dim(nl_base_daily_rets_longshort)[1], ]


l_rl_daily_rets_longshort <- read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR/Linear_Shrinkage/results/model_linear_daily_returns.csv",
                                    sep = ",")
l_base_daily_rets_longshort <-  read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR/Linear_Shrinkage/results/cov1para_daily_returns.csv",
                                           sep = ",")


for (colname in colnames(nl_rl_daily_rets_longshort)[2:6]){
  rets = data.frame(
    as.numeric(nl_rl_daily_rets_longshort[, colname]),
    as.numeric(nl_base_daily_rets_longshort[, colname])
    )
  out <- hac.inference.log.var(rets)
  print(paste0("p-values for NL Long Short p", colname, ":"))
  print(out$p.Values)
}

for (colname in colnames(l_base_daily_rets_longshort)[2:6]){
  rets = data.frame(
    as.numeric(l_rl_daily_rets_longshort[, colname]),
    as.numeric(l_base_daily_rets_longshort[, colname])
  )
  out <- hac.inference.log.var(rets)
  print(paste0("p-values for L Long Short p", colname, ":"))
  print(out$p.Values)
}




# LONG ONLY
nl_rl_daily_rets_long <- read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR_long_only/NonLinear_Shrinkage/results/model_NL_daily_returns.csv",
                                        sep = ",")
nl_base_daily_rets_long <-  read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR_long_only/NonLinear_Shrinkage/results/QIS_base_daily_returns.csv",
                                           sep = ",")
nl_base_daily_rets_long <- nl_base_daily_rets_long[5041:dim(nl_base_daily_rets_long)[1], ]


l_rl_daily_rets_long <- read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR_long_only/Linear_Shrinkage/results/model_linear_daily_returns.csv",
                                       sep = ",")
l_base_daily_rets_long <-  read.csv2("H:/all/RL_Shrinkage_2024/ONE_YR_long_only/Linear_Shrinkage/results/cov1para_daily_returns.csv",
                                          sep = ",")


for (colname in colnames(nl_rl_daily_rets_long)[2:6]){
  rets = data.frame(
    as.numeric(nl_rl_daily_rets_long[, colname]),
    as.numeric(nl_base_daily_rets_long[, colname])
  )
  out <- hac.inference.log.var(rets)
  print(paste0("p-values for NL Long Only p", colname, ":"))
  print(out$p.Values)
}

for (colname in colnames(l_base_daily_rets_long)[2:6]){
  rets = data.frame(
    as.numeric(l_rl_daily_rets_long[, colname]),
    as.numeric(l_base_daily_rets_long[, colname])
  )
  out <- hac.inference.log.var(rets)
  print(paste0("p-values for L Long Only p", colname, ":"))
  print(out$p.Values)
}

