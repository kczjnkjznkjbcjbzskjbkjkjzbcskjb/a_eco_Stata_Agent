*==================================================
* 回归分析：price 与 mpg、weight 的关系
* 数据集：auto.dta
*==================================================

clear all
set more off

* 加载 auto 数据集
sysuse auto, clear

* 查看数据基本信息
describe
summarize price mpg weight

* 回归分析：price = β0 + β1*mpg + β2*weight + ε
regress price mpg weight

* 输出详细的回归结果
estimates store model1
estimates table model1, star stats(N r2 r2_a)

* 结束
exit