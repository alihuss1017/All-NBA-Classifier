# SUMMARY
The program aims to predict whether or not an NBA player would make an All-NBA team(1st, 2nd, or 3rd) dependent on their stats in the regular season. Implements the use of an SVM and logistic regression algorithm, comparing the results between the two using evaluation metrics such as precision and recall. 

## STATS USED 

PTS (Points)

TRB (Total Rebounds)

AST (Assists)

%GPnSround (Percentage of Games Played and Started in Decimal Form)

PER (Player Efficiency Rating)

WS (Win Shares)

BPM (Box Plus/Minus)

VORP (Value Over Replacement Player)

All stats used are from basketball-reference.com

## PREDICTION METRICS USED

TN(True Negative): How many NBA players were correctly predicted to not make an all-NBA team(y=0)?

TP(True Positive): How many NBA players were correctly predicted to make an all-NBA team(y=1)?

FN(False Negative): How many NBA players were incorrectly predicted to not make an all-NBA team?

FP(False Positive): How many NBA players were incorrectly predicted to not make an all-NBA team?

## EVALUATION METRICS USED

Precision: TP/(TP+FP). Of all NBA players predicted to make an all-NBA team, what fraction actually made an all-NBA team?

Recall: TP/(TP+FN). Of all NBA players who made an all-NBA team, what fraction was correctly predicted to make an all-NBA team?

f1Score: 2(Precision*Recall)/(Precision+Recall). An f1Score is more indicative of the algorithm's performance rather than the accuracy score in this situation since the number of players correctly predicted to not make an all-NBA team(true negatives) is by far the greatest value amongst the other prediction metrics(true positives, false negatives, false positives), and thus every accuracy score would be in the high 90's, not reflective of the algorithm's performance.

Accuracy Score:(TP+TN)/(TP+TN+FP+FN). Of all NBA players in the testing data, how many were correctly predicted to make/not make an all-NBA team?

###  SVM ALGORITHM PERFORMANCE
![svm](https://user-images.githubusercontent.com/83521645/132613508-42635e19-cc21-4ae3-a3ba-193c716a627c.jpg)

Shown above is the result of the testing data(performed on SVM algorithm) that includes a player who was either predicted to make an all-NBA team or actually made an all-NBA team. NOTE: Player names may appear more than once, this means that the testing data contained multiple seasons from a single player.

#### EVALUATION RESULTS
True Negatives: 308

True Positives: 22

False Negatives: 2 (DeMarcus Cousins, Joel Embiid)

False Positives: 3 (Bradley Beal, Zion Williamson, Anthony Davis)

Precison: 0.88

Recall: 0.92

f1Score: 0.90

Accuracy Score: 0.99




###### POINTS AND ALL-NBA CORRELATION



![ptsSVM](https://user-images.githubusercontent.com/83521645/132613641-b40eab49-c255-4cb8-91f5-650b75d427e2.png)

This is a 3D scatter plot where the data points are all players who were either predicted to make an all-NBA team or actually made an all-NBA team. There are two independent variables, the percentage of games played and started(%GPnS), and points(PTS). The dependent variable is whether a player would make an all-NBA team, and therefore, the only value y can be is 0(NO) or 1(YES). The correct predictions are precisely located where the actual predictions are. There is a strong positive correlation between points and players making on all-NBA team.

###### BRIEF GRAPHICAL EXPLANATION

![Screenshot (8)](https://user-images.githubusercontent.com/83521645/132623362-ff67af67-0fdb-4313-a24f-1d672b0a158f.png)

Looking at the 3D graph from a top view, it is shown that the blue and orange points do indeed completely overlap each other when the prediction is correct. However, if there is an orange data point that partially overlaps a blue data point, this would indicate that there was a false prediction for that player. As seen in the PTS graph above, although it is difficult to visualize, there are to be multiple points where this is the case.

###### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perSVM](https://user-images.githubusercontent.com/83521645/132613715-6363d2c6-112a-4fa0-817c-08c52984a0ae.png)

This graph's independent variables are PER and %GPnS. There is a positive correlation between %GPnS and making an all-NBA team, and PER does not seem to influence the outcome. There is a relatively strong, positve correlation between PER and %GPnS with making an all-NBA team.

###### WIN SHARES AND ALL-NBA CORRELATION

![wsSVM](https://user-images.githubusercontent.com/83521645/132613748-d1626f0a-623c-4d9f-bb3c-804661dcc496.png)
This graph's independent variables are WS and %GPnS. There is a relatively strong, positive correlation between WS and %GPnS with making an all-NBA team.

###### BOX PLUS/MINUS AND ALL-NBA CORRELATION

![bpmSVM](https://user-images.githubusercontent.com/83521645/132613754-50794b21-dd28-4791-9f1b-e13bf95068cf.png)
This graph's independent variables are BPM and %GPnS. There is a positive correlation between BPM and %GPnS with making an all-NBA team.

###### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION

![vorpSVM](https://user-images.githubusercontent.com/83521645/132613763-63540d32-eb4e-4e0b-a1e6-da0b7c47e509.png)
This graph's independent variables are VORP and %GPnS. There is a positive correlation between %GPnS and making an all-NBA team, while there is a relatively weak, positive correlation between VORP and making an all-NBA team.

###### DATA ACCURACY RESULTS
![resultSVM](https://user-images.githubusercontent.com/83521645/132613773-837dbf35-cfeb-4d00-9ed5-055480b8dc89.png)

This bar graph looks at the accuracy of the algorithm, comparing true positives, false negatives, and false positives. As shown, the algorithm correctly predicted most of the players that actually made an all-NBA team.

#### PRECISION AND RECALL RESULTS
![pnrSVM](https://user-images.githubusercontent.com/83521645/132613782-3ba00019-4062-45cc-a5ec-885a1f1e7a14.png)

This bar graph compares the magnitudes of precision and recall. In this case the recall was slightly higher, meaning the algorithm was more lenient in predicting players to make an all-NBA team.






### LOGISTIC REGRESSION ALGORITHM PERFORMANCE
![logregData](https://user-images.githubusercontent.com/83521645/132617448-2d25c1c8-5545-4bc5-af25-bf30fa4012ca.jpg)

Shown above is the result of the testing data(performed on logistic regression algorithm) that includes a player who was either predicted to make an all-NBA team or actually made an all-NBA team. NOTE: Player names may appear more than once, this means that the testing data contained multiple seasons from a single player.

#### EVALUATION RESULTS
True Negatives: 314

True Positives: 13

False Negatives: 8 (Paul George, Chris Paul, Rudy Gobert, Draymond Green, Kawhi Leonard, DeMarcus Cousins, Jimmy Butler, Joel Embiid)

False Positives: 0

Precision: 1.0

Recall: 0.62

f1Score: 0.76

Accuracy Score: 0.98

###### POINTS AND ALL-NBA CORRELATION
![ptsLR](https://user-images.githubusercontent.com/83521645/132617480-53a3c66c-5cfd-47cd-af45-c80d6aebffad.png)

There are two independent variables, the percentage of games played and started(%GPnS), and points(PTS). There is a strong positive correlation between points and players making on all-NBA team.

###### PLAYER EFFICIENCY RATING AND ALL-NBA CORRELATION
![perLR](https://user-images.githubusercontent.com/83521645/132617488-ff60ac4d-479f-4712-b9ab-693644c5305a.png)

There are two independent variables, PER and %GPnS. There is a  positive correlation between PER and %GPnS with players making on all-NBA team.

###### WIN SHARES AND ALL-NBA CORRELATION
![wsLR](https://user-images.githubusercontent.com/83521645/132617498-efa68682-5970-4e79-9466-c9beb8ed6b00.png)

There are two independent variables, WS and %GPnS. There is a strong positive correlation between WS and GPnS with players making on all-NBA team.

###### BOX PLUS/MINUS AND ALL-NBA CORRELATION
![bpmLR](https://user-images.githubusercontent.com/83521645/132617503-9c91e675-957f-4c58-9f0c-7ec1e369ba5c.png)

There are two independent variables, BPM and %GPnS. There is a positive correlation between BPM and %GPnS with players making on all-NBA team.

###### VALUE OVER REPLACEMENT PLAYER AND ALL-NBA CORRELATION
![vorpLR](https://user-images.githubusercontent.com/83521645/132617510-c1000cbb-84db-4529-b036-9f76850eaaa4.png)

There are two independent variables, VORP and %GPnS. There is a strong positive correlation between VORP and %GPnS with players making on all-NBA team.

###### DATA ACCURACY RESULTS
![resultLR](https://user-images.githubusercontent.com/83521645/132617532-33602f3d-f03c-4862-82ce-4455ee0b6e16.png)

This bar graph looks at the accuracy of the algorithm, comparing true positives, false negatives, and false positives. As shown, the algorithm correctly predicted more than half of the players to make an all-NBA team.

###### PRECISION AND RECALL RESULTS
![pnrLR](https://user-images.githubusercontent.com/83521645/132617538-02444d02-77dd-4892-82f3-5d4870b6930d.png)

This bar graph compares the magnitudes of precision and recall. In this case the precision was much greater, indicating that this algorithm was strongly selective in predicting players to make an all-NBA team.

## ANALYSIS AND CONCLUSION

![f1Score](https://user-images.githubusercontent.com/83521645/132939604-d5422706-22a9-4e13-adf9-9d16149150df.png)


The f1Score and accuracy score of the SVM algorithm was generally greater than that of the logistic regression algorithm, and therefore, the SVM algorithm performed best at predicting whether or not an NBA player would make an all-NBA team. However, both algorithms typically have an f1Score no lower than 0.7, indicating accurate results. There are many adjustments to be made to the program, including better visualization graphics and hyperparameter tuning.