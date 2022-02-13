# Titanic Survival Prediction 
### Historical Backgroud
* Titanic was a British passenger liner that sank in the North Atlantic Ocean on 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City.
* Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew

### Problem definition
What sorts of people were more likely to survive the Titanic sinking?

### Dataset 

| Feature  | Description                                                          |
|----------|----------------------------------------------------------------------|
| Survival | Survival (0 = No; 1 = Yes)                                           |
| Pclass   | Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)                          |
| Name     | Full name                                                            |
| Sex      | Sex                                                                  |
| Age      | Age                                                                  |
| Sibsp    | Number of Siblings/Spouses Aboard                                    |
| Parch    | Number of Parents/Children Aboard                                    |
| Ticket   | Ticket Number                                                        |
| Fare     | Passenger Fare                                                       |
| Cabin    | Cabin                                                                |
| Embarked | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) |

### Summary
#### Fields

<img width="1000" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/Survival%20Rate.png">

<img width="700" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/Survival%20rate%20with%20and%20without%20family.png">

#### Simple Models
As we can see below, the three categorical features Sex, Class and has_family, explain most of the variation in y_label:
1. More than 90% of the women from 1st and 2nd classess survived
2. ~85% of men from 2nd and third classess didn't survived, and ~90% of them that travel alone died. 

This means that we can create a simple prediction with ~90% accuracy for at least the ~60% of the passangers following under conditions (1) or (2)
<img width="700" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/Survival%20rate%20across%20the%20three%20categorical%20features.png">

### Training 
I used CV random search to tune the optimal parameters for the model. I defined a large grid space of possible parameters and sample it randomly. Here is how I handle the different parametrs: 

#### Random Forest
Forest Design:
* **n_estimators** - the loss (in train and test) converges to the same value as a function of the number of trees. Hence, I chose to use one large number of trees (e.g. 250 to a dataset of size 900), and not try out mulitple different values.
* **max_depth** - the depth of the trees is choosen using random search. Larger depth will exponentially incrsrease the complexity of the model (e.g. at depth of log_2 ( n ) for n number of samples, we would have 100% accuracy in during training ). 
* **min_samples_split**, **min_samples_leaf** - other ways to restrict the size of the trees. In addition to searching for different values of max_depth, we will randomlly search for different design for the trees in those parametres. If we use high restriction on the size the leaf, we can use deeper depth and vice versa. 
<img width="1500" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/rf_scores.png">


#### Logistic Regression 
I used lasso, ridge and simple-linear.
<img width="700" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/lr.png">

#### SVM
poly, rbf and simple-linear.


<img width="700" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/SVM_search_results.png">

#### KNN
* **n_neighbors** - number of neighbors to use. The more used the lower the variance and the higehr the bias. If we use n=number of data points, we get a uniform prediction, and if we use n=1 we get highly variable prediction. 
* **weights** - uniform weights will predicts the average of the neighbors, whereas "distance" option will take the distance-weighted-average of the neighbors. If we used distance-weighted average, the number of neighbors parametrs will have a reduced effect on prediction. 

<img width="1500" alt="data_map" src="https://github.com/hiladar0/titanic/blob/main/images/knn_search_results.png">



