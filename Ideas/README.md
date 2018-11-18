# Ideas

> This file is for ideas recording during discussion or online searching

<br>

## Paper Work

* [Modeling MOOC Dropouts](https://pdfs.semanticscholar.org/1bc9/58caeab3036836945a2278ba37721d4cab1e.pdf) (**Seems useful**)
  * [code implementation](https://bitbucket.org/lics229/mooc-dropout-prediction/overview)
  * Features
    * Basic Counts
    * Ratio (item visited/number of courses)
    * Number of activities on Day X (30 days in total)
    * Activities on Weekdays and hours (Study patterns)
    * Study sessions (set min_spans and find the sessions studied in one week)
    * Failed features: class volume, parallel enrollment, frequent sequences event...
  * Methods
    * Gradient Boosting, Random Forest, AdaBoost, Logistic regression, Linear SVM
* [Predicting MOOC Dropout over Weeks Using Machine Learning Method](http://www.aclweb.org/anthology/W14-4111)
  * Number of requests: total number of requests including page views and video click actions
  * Number of sessions: number of sessions is supposed to be a reflection of high engagement, because more sessions indicate more often logging into the learning platform
  * Number of active days: we define a day as an active day if the student had at least one session on that day
  * Number of page views: the page views include lecture pages, wiki pages, homework pages and forum pages
  * Number of page views per session: the average number of pages viewed by each participant per session
  * Number of video views: total number of video click actions
  * ......(a lot of) 
* [Predicting Attrition Along the Way: The UIUC Model](https://www.aclweb.org/anthology/W/W14/W14-4110.pdf)
* [Dropout prediction in MOOCs using behavior features and multi-view semi-supervised learning](https://ieeexplore.ieee.org/document/7727598)
  * Counts of behaviors per week
* [The story of how I built hundreds of predictive models….](https://www.r-bloggers.com/kdd-cup-2015-the-story-of-how-i-built-hundreds-of-predictive-models-and-got-so-close-yet-so-far-away-from-1st-place/)
  * Number of courses per student take
  * Number of times each particular event was logged for every enrollment_id
  * Average timestamp for each event for each enrollment_id
  * Min and Max timestamp for each event for each enrollment_id
  * Total time elapsed from the first to the last instance of each event for each enrollment_id
  * Overall average timestamp for each enrollment_id
  * count of course components by course_id
  * the number of ‘children’ per course_id
* [Predicting Dropouts in MOOC’S (IJIR)](https://www.onlinejournal.in/IJIRV2I6/073.pdf)
* 

