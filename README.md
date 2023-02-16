# joint data-model uncertainty modeling

In this study, the input data uncertainty, target data uncertainty and model uncertainty are jointly modeled in a deep learning precipitation forecasting framework to estimate the predictive uncertainty. Specifically, the data uncertainty is estimated a priori and the input uncertainty is propagated forward through model weights according to the law of error propagation. The model uncertainty is considered by sampling from the parameters and is coupled with input and target data uncertainties in the objective function during the training process. Finally, the predictive uncertainty is produced by propagating the input uncertainty in the testing process.

![image](https://user-images.githubusercontent.com/16514945/219365515-000f604b-b806-4992-b5d8-9f6bc2c2bfb5.png)

Please refer to the following paper for more information:

Xu, L., Chen, N., Yang, C., Yu, H., & Chen, Z. (2022). Quantifying the uncertainty of precipitation forecasting using probabilistic deep learning. Hydrology and Earth System Sciences, 26(11), 2923-2938.
