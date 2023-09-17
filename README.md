# SUMMARY
The program aims to predict whether or not an NBA player would make an All-NBA team(1st, 2nd, or 3rd) dependent on their stats in the regular season. Implements the use of an SVM and logistic regression algorithm, comparing the results between the two using evaluation metrics such as precision and recall. 

## DATASET STATS
G (Games Played) 

GPnS% (Percentage of games started from total played) 

GPnSround% (Percentage of games started from total played, rounded)

GS (Games Started)

MP (Minutes Played Per Game)

FG (Field Goals Per Game)

FGA (Field Goals Attempted Per Game)

FG% (Field Goal Percentage)

3P (3 Pointers Made Per Game)

3PA (3 Pointers Attempted Per Game)

3P% (3 Pointer Percentage)

2P (2 Pointers Made Per Game)

2PA (2 Pointers Attempted Per Game)

2P% (2 Pointer Percentage)

eFG% (Efficiency Field Goal Percentage)

TRB (Total Rebounds Per Game)

AST (Assists Per Game)

STL (Steals Per Game)

BLK (Blocks Per Game)

TOV (Turnovers Per Game)

PF (Personal Fouls Per Game)

PTS (Points Per Game)

PER (Player Efficiency Rating)

WS (Win Shares)

BPM (Box Plus/Minus)

VORP (Value Over Replacement Player)

All-NBA? (Whether or not a player makes an All-NBA team)

All stats used are from basketball-reference.com

## STATS USED IN MODEL

GS

FG

FGA

2P

2PA

AST

TOV
 
PTS
 
WS
 
VORP
 
All-NBA?

## GROUND TRUTHS

TN(True Negative): How many NBA players were correctly predicted to not make an All-NBA team(y=0)?

TP(True Positive): How many NBA players were correctly predicted to make an All-NBA team(y=1)?

FN(False Negative): How many NBA players were incorrectly predicted to not make an All-NBA team?

FP(False Positive): How many NBA players were incorrectly predicted to not make an All-NBA team?

## EVALUATION METRICS USED

Precision: 
$$ \frac{TP}{TP+FP} $$
 Of all NBA players predicted to make an all-NBA team, how many actually made an all-NBA team?

Recall: 
$$ \frac{TP}{TP + FN} $$
 Of all NBA players who made an all-NBA team, how many were correctly predicted to make an all-NBA team?

f1Score: 
$$ \frac{2 * Precision * Recall}{Precision + Recall} $$
 An f1Score is more indicative of the algorithm's performance rather than the accuracy score in this situation since the number of players correctly predicted to not make an all-NBA team(true negatives) is by far the greatest value amongst the other prediction metrics(true positives, false negatives, false positives), and thus every accuracy score would be in the high 90's, not reflective of the algorithm's performance.

Accuracy Score: 
$$ \frac{TP+TN}{TP+TN+FP+FN} $$
Of all NBA players in the testing data, how many were correctly predicted to make/not make an all-NBA team?