# A2-Classification-with-SVM-BP-and-MLR



Contenido
General information	2
Support Vector Machines (SVM), using free software	2
Dataset Ring	2
Dataset bank	9
Dataset captura (.PCAP)	14
Back-Propagation (BP), using free software	18
Dataset Ring	18
Dataset bank	22
Multiple Linear Regression (MLR), using free software	23
Dataset Ring	23
Dataset bank	30
Dataset captura (.PCAP)	31

 
General information
Github repository: https://github.com/Cod3Frank/A2-Classification-with-SVM-BP-and-MLR

Student name: Francisco Campos Batista
Support Vector Machines (SVM), using free software 
Support Vector Machines (SVM) are a type of supervised machine learning algorithm used for classification and regression tasks. The primary objective of SVM is to find a hyperplane in an N-dimensional space (where N is the number of features) that distinctly classifies data points into different classes. SVM is particularly effective in high-dimensional spaces and is widely used in various fields, including image classification, text categorization, and bioinformatics.
SVMs have proven to be effective in various applications, and their versatility, especially with the use of different kernel functions, makes them a popular choice in machine learning. However, SVMs might not perform as well on very large datasets, and their training can be computationally expensive. Additionally, the choice of the appropriate kernel and tuning parameters is crucial for achieving good performance.
Dataset Ring
Training
 
Test
 
Cross validation using k-fold cross-validation
K-fold ---------------------------------------------------------------
Model accuracy: 0.9724
 
Confusion matrix:
 [[1232   68]
 [   1 1199]]
Percentage classification error obtained from validation set: 2.76%
K-fold ---------------------------------------------------------------
Model accuracy: 0.974
 
Confusion matrix:
 [[1254   61]
 [   4 1181]]
Percentage classification error obtained from validation set: 2.6%
K-fold ---------------------------------------------------------------
Model accuracy: 0.9688
 
Confusion matrix:
 [[1241   76]
 [   2 1181]]
Percentage classification error obtained from validation set: 3.1199999999999997%
K-fold ---------------------------------------------------------------
Model accuracy: 0.9708
 
Confusion matrix:
 [[1202   69]
 [   4 1225]]
Percentage classification error obtained from validation set: 2.92%
Mean percentage classification error obtained from cross validation: 2.8499999999999996%
Training
 
Model accuracy: 0.9764
 
Confusion matrix:
 [[5108  225]
 [  11 4656]]
Percentage classification error obtained from test set: 2.36%

Interpretation of the data
The value "Model accuracy: 0.9764" indicates that the model correctly classified 97.64% of instances in the test set. This 97.64% accuracy suggests a solid performance of the model in the specific classification task for which it was trained.
In general, a high accuracy is a positive indicator, suggesting that the model is effective in classifying instances. However, it's always important to consider the context of the problem and examine other metrics, especially if there are asymmetric costs associated with specific classification errors.
In summary, a high accuracy value indicates that the model is doing well in classification, but it is always advisable to evaluate other metrics and consider the specific context of the problem.
Confusion Matrix:
True Positives (TP): 5108
False Positives (FP): 225
False Negatives (FN): 11
True Negatives (TN): 4656

Interpretation:
The model correctly predicted 5108 instances of the positive class.
There were 225 instances that the model predicted as positive but were actually negative.
Only 11 instances were incorrectly classified as negative when they were positive.
The model correctly predicted 4656 instances of the negative class.

Percentage Classification Error:
The percentage classification error obtained from the test set is 2.36%. This indicates the proportion of instances in which the model made incorrect predictions relative to the total number of instances.

In summary, the confusion matrix provides a detailed breakdown of the model's predictions, and the low percentage classification error (2.36%) suggests that the model has a strong performance on the test set. The model particularly excels in correctly predicting instances of the positive and negative classes.
 
Dataset bank
K-fold ---------------------------------------------------------------
Model accuracy: 0.9065533980582524
 
Confusion matrix:
 [[723  12]
 [ 65  24]]
Percentage classification error obtained from validation set: 9.344660194174757%
K-fold ---------------------------------------------------------------
Model accuracy: 0.9053398058252428
 
Confusion matrix:
 [[733   2]
 [ 76  13]]
Percentage classification error obtained from validation set: 9.466019417475728%
K-fold ---------------------------------------------------------------
Model accuracy: 0.8956310679611651
 
Confusion matrix:
 [[716   7]
 [ 79  22]]
Percentage classification error obtained from validation set: 10.436893203883495%
K-fold ---------------------------------------------------------------
Model accuracy: 0.905224787363305
 
Confusion matrix:
 [[735   7]
 [ 71  10]]
Percentage classification error obtained from validation set: 9.477521263669502%
Mean percentage classification error obtained from cross validation: 9.68127351980087%
Training
Model accuracy: 0.9029126213592233
 
Confusion matrix:
 [[724   9]
 [ 71  20]]
Percentage classification error obtained from test set: 9.70873786407767%
•	Model Accuracy: 90.29%
Accuracy is a measure of the overall correctness of the model's predictions. In this case, the model correctly predicts the class of instances approximately 90.29% of the time.
•	Confusion matrix
True Positives (TP): 20 instances were correctly classified as positive.
False Positives (FP): 9 instances were incorrectly classified as positive.
True Negatives (TN): 724 instances were correctly classified as negative.
False Negatives (FN): 71 instances were incorrectly classified as negative.
•	Percentage Classification Error:
Percentage Classification Error: 9.71% The model has a high accuracy of 90.29%, and the confusion matrix provides a detailed breakdown of true positives, false positives, true negatives, and false negatives. The low percentage classification error indicates that the model performs well on the test set, with only around 9.71% misclassified instances.
Dataset captura (.PCAP)
K-fold ---------------------------------------------------------------
Model accuracy: 1.0
 
Confusion matrix:
 [[2 0]
 [0 7]]
Percentage classification error obtained from validation set: 0.0%
K-fold ---------------------------------------------------------------
Model accuracy: 0.8888888888888888
 
Confusion matrix:
 [[6 1]
 [0 2]]
Percentage classification error obtained from validation set: 11.11111111111111%
K-fold ---------------------------------------------------------------
Model accuracy: 1.0
 
Confusion matrix:
 [[4 0]
 [0 5]]
Percentage classification error obtained from validation set: 0.0%
K-fold ---------------------------------------------------------------
Model accuracy: 0.8888888888888888
 
Confusion matrix:
 [[5 0]
 [1 3]]
Percentage classification error obtained from validation set: 11.11111111111111%
Mean percentage classification error obtained from cross validation: 5.555555555555555%
Test
Model accuracy: 1.0
 
Confusion matrix:
 [[6 0]
 [0 3]]
Percentage classification error obtained from test set: 0.0%

•  Model Accuracy: 1.0 (100%)
•	This indicates that your model achieved perfect accuracy on the test set. All predictions made by the model were correct, suggesting that it performed exceptionally well.
•  Confusion Matrix:
lua
The confusion matrix is a 2x2 matrix used for binary classification. In this case:
True Positive (TP): 6 instances
False Positive (FP): 0 instances
True Negative (TN): 3 instances
False Negative (FN): 0 instances
The absence of any false positives or false negatives reinforces the 100% accuracy mentioned earlier.
•	Percentage Classification Error: 0.0%
This is calculated as (FP + FN) / Total instances. In your case, it's (0 + 0) / (6 + 0 + 0 + 3) = 0.0%. This further confirms the perfect performance of your model, as there are no misclassifications.
In summary, the model appears to be highly accurate, achieving a perfect score on the test set with no misclassifications. While these results are impressive, it's essential to consider the possibility of overfitting or other factors that might affect the model's generalization to new data. Additionally, it's always good practice to evaluate models on diverse datasets to ensure robust performance.
Back-Propagation (BP), using free software 

Dataset Ring
Training
 
Test
 

Training || TensorFlow
 
Test || Keras

 
 
Confusion matrix:
 [[5333    0]
 [4667    0]]
Percentage classification error obtained from test set: 46.67%
1.	
o	This confusion matrix indicates a binary classification problem. Breaking it down:
	True Positives (TP): 5333 instances
	False Positives (FP): 0 instances
	True Negatives (TN): 0 instances
	False Negatives (FN): 4667 instances
Percentage Classification Error: 46.67%
o	The percentage classification error is calculated as (FP + FN) / Total instances. In this case, it's (0 + 4667) / (5333 + 0 + 0 + 4667) = 46.67%.
Analysis:
The model seems to be only predicting the positive class (class 1) and not identifying any instances of the negative class (class 0). This is evident from the zeros in the TN and FP cells of the confusion matrix.
The percentage classification error of 46.67% suggests that almost half of the instances are misclassified.
Dataset bank


 
Confusion matrix:
 [[688  44]
 [ 52  40]]
Percentage classification error obtained from test set: 11.650485436893204%
True Positives (TP): 42 instances were correctly classified as positive.
False Positives (FP): 47 instances were incorrectly classified as positive.
True Negatives (TN): 685 instances were correctly classified as negative.
False Negatives (FN): 50 instances were incorrectly classified as negative.
Percentage Classification Error: 11.77%
This indicates that approximately 11.77% of instances in the test set were misclassified.
Summary:
The confusion matrix provides a detailed breakdown of true positives, false positives, true negatives, and false negatives. The model has a percentage classification error of 11.77%, suggesting that around 11.77% of instances in the test set were misclassified. Further analysis or exploration of other evaluation metrics may be beneficial to gain a comprehensive understanding of the model's performance.

Multiple Linear Regression (MLR), using free software 

Dataset Ring
Training
 
Test
 
Cross validation
Training for fold 1 ...
Model accuracy: 0.4448
 
Confusion matrix:
 [[1112  188]
 [1200    0]]
Percentage classification error obtained from validation set: 55.52%
------------------------------------------------------------------------
Training for fold 2 ...
Model accuracy: 0.526
 
Confusion matrix:
 [[1315    0]
 [1185    0]]
Percentage classification error obtained from validation set: 47.4%
------------------------------------------------------------------------
Training for fold 3 ...
Model accuracy: 0.4584
 
Confusion matrix:
 [[1146  171]
 [1183    0]]
Percentage classification error obtained from validation set: 54.16%
------------------------------------------------------------------------
Training for fold 4 ...
Model accuracy: 0.5084
 
Confusion matrix:
 [[1271    0]
 [1229    0]]
Percentage classification error obtained from validation set: 49.16%

Results of cross validation:
------------------------------------------------------------------------
Percentage classification error obtained from validation set per fold
------------------------------------------------------------------------
> Fold 1 - 55.52%
------------------------------------------------------------------------
> Fold 2 - 47.4%
------------------------------------------------------------------------
> Fold 3 - 54.16%
------------------------------------------------------------------------
> Fold 4 - 49.16%
------------------------------------------------------------------------
Mean percentage classification error obtained from cross validation:
> 51.559999999999995% (+- 3.372951230006151)
------------------------------------------------------------------------
Test
 
Model accuracy: 0.5289
 
Confusion matrix:
 [[5289   44]
 [4667    0]]
Percentage classification error obtained from test set: 47.11%
•	Model Accuracy: 0.5289 (52.89%)
Accuracy is a measure of the overall correctness of the model's predictions. In this case, the model correctly predicts the class of instances approximately 52.89% of the time.
True Positives (TP): 0 instances were correctly classified as positive.
False Positives (FP): 44 instances were incorrectly classified as positive.
True Negatives (TN): 5289 instances were correctly classified as negative.
False Negatives (FN): 4667 instances were incorrectly classified as negative.
•	Percentage Classification Error:
Percentage Classification Error: 47.11%
The model has an accuracy of 52.89%, and the confusion matrix highlights the distribution of correct and incorrect predictions. The percentage classification error provides additional insight into the overall model performance on the test set. Further analysis may be needed to understand specific challenges and areas for improvement.
Dataset bank
Model accuracy: 0.8932038834951457
 
 
Confusion matrix:
 [[700   9]
 [ 79  36]]
Percentage classification error obtained from test set: 10.679611650485436%
Model Accuracy: 89.32% Accuracy is a measure of the overall correctness of the model's predictions. In this case, the model correctly predicts the class of instances approximately 89.32% of the time.
Confusion Matrix:
True Positives (TP): 36 instances were correctly classified as positive.
False Positives (FP): 9 instances were incorrectly classified as positive.
True Negatives (TN): 700 instances were correctly classified as negative.
False Negatives (FN): 79 instances were incorrectly classified as negative.
Percentage Classification Error: Percentage Classification Error: 10.68%
The model has a good accuracy of 89.32%, and the confusion matrix provides a detailed breakdown of true positives, false positives, true negatives, and false negatives. The percentage classification error is relatively low, suggesting that the model performs well on the test set with only around 10.68% misclassified instances.
Dataset captura (.PCAP)
Model accuracy: 0.0
 
 
Model accuracy: 0.0

Confusion matrix:
 [[0 2 0 0]
 [0 0 0 0]
 [0 1 0 0]
 [0 6 0 0]]
Percentage classification error obtained from test set: 100.0%

•  Model Accuracy: 0.0 (0%)
•	An accuracy of 0% indicates that the model didn't make any correct predictions on the test set. This could suggest a significant issue with the model's performance.
•	The confusion matrix is a 4x4 matrix, suggesting a multi-class classification problem. In this case:
o	True Positives (TP): 0 instances
o	False Positives (FP): 2 instances
o	True Negatives (TN): 0 instances
o	False Negatives (FN): 7 instances
•	It's notable that there are no correct predictions (TP and TN) based on the zeros in the matrix.
•  Percentage Classification Error: 100.0%
•	This indicates that all instances in the test set were misclassified. The percentage classification error is calculated as (FP + FN) / Total instances. In your case, it's (2 + 7) / (0 + 2 + 0 + 0 + 0 + 0 + 1 + 0 + 6 + 0) = 100.0%.
•	
In summary, the provided model has a critical issue, as it didn't make any correct predictions on the test set, resulting in a 0% accuracy and a 100% classification error. This suggests a need for thorough investigation into the model architecture, training process, and data quality to identify and address the underlying problems.
