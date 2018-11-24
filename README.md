# Drop-MOOC-Prediction

>  COMP4331 Data Mining Group Project



* 1st round: event_count

```
             precision    recall  f1-score   support

          0       0.70      0.51      0.59      4902
          1       0.88      0.94      0.91     19111

avg / total       0.84      0.85      0.85     24013
```

<br>

* 2nd round: add weekly_event_count

```
 precision    recall  f1-score   support

          0       0.74      0.57      0.64      4902
          1       0.90      0.95      0.92     19111

avg / total       0.86      0.87      0.86     24013
```
<br>

* 3rd round: add total and weekly session_count

  (RF Forest seems meet the bottleneck. Though new features are important in feature importance, the report is exactly the same as above)

 ```
	 {'access_count': 0.15046517430736295,
     'discussion_count': 0.037059672625547616,
     'navigate_count': 0.09322651948261372,
     'page_close_count': 0.10582247960769997,
     'problem_count': 0.07999196829596014,
     'video_count': 0.05923997479899466,
     'wiki_count': 0.021377316802125693,
     'week_one_event': 0.026758710091908024,
     'week_two_event': 0.03845727813902135,
     'week_three_event': 0.037540751006407094,
     'week_four_event': 0.06167680930542194,
     'week_five_event': 0.0773541997062687,
     'week_six_event': 0.006393035384710254,
     'week_one_session': 0.017295390978427222,
     'week_two_session': 0.02432480102871223,
     'week_three_session': 0.026945581744095313,
     'week_four_session': 0.05026792788345937,
     'week_five_session': 0.08154207954129963,
     'week_six_session': 0.004260329269964542}
 ```
