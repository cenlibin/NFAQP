# NFAQP

This repo is the implementation of NFAQP, a normalizing flow-based AQP approach. Our Paper "Normalizing FLow based Approximate Query Processing" is accepted by [Advanced Data Mining and Applications (ADMA) 2023](https://adma2023.uqcloud.net)

## Abstract
With the unprecedented rate at which data is being generated, Approximate Query Processing (AQP) techniques are widely demanded in various areas. Recently, machine learning techniques have made remarkable progress in this field. However, data with large domain sizes still cannot be handled efficiently by existing approach. Besides, the accuracy of the estimate is easily affected by the number of predicates, which may lead to erroneous decisions for users in complex scenarios.

In this paper, we propose NFAQP, a novel AQP approach that leverages normalizing flow to efficiently model the data distribution and estimate the aggregation function by multidimensional Monte Carlo integration. Our model is highly lightweight - often just a few dozen of KB - and is unaffected by large domains. More importantly, even under queries with a large number of predicates, NFAQP still achieves relatively low approximation errors. Extensive experiments conducted on three real-world datasets demonstrate that NFAQP outperforms baseline approaches in terms of accuracy and model size, while maintaining relatively low latency.
![The Framework of NFAQP](./imgs/framework.png)