import pandas as pd
import numpy as np 
import time
from datetime import datetime
from load_data import Course_Date
from load_data import Course_Object
from load_data import Log
from load_data import Enrollment
from load_data import Truth


class Preprocessor():
    def __init__(self, log_path, enroll_path, truth_path):
        self.course_date = Course_Date('data/date.csv')
        self.course_obj = Course_Object('data/object.csv')
        self.log = Log(log_path)
        self.enroll = Enrollment(enroll_path)
        self.truth = Truth(truth_path)

    def get_truth(self):
        return self.truth
    
    def get_merge_df(self):
        enroll_df = self.enroll.get_data()
        log_df = self.log.get_data()

        course_date_df = pd.DataFrame.from_dict(self.course_date.get_course_info()).T
        course_date_df.reset_index(inplace = True)
        course_date_df.rename(columns={"index": "course_id"}, inplace = True)

        enroll_course_df = enroll_df.merge(course_date_df, on='course_id', how='inner')
        log_enroll_course_df = log_df.merge(enroll_course_df, on = 'enrollment_id', how='inner')

        return log_enroll_course_df

    def weekly_event_count(self):
        start_time = time.time()

        merge_df = self.get_merge_df().filter(items=['enrollment_id', 'time', 'event', 'start'])
        merge_df= merge_df.assign(week_index = merge_df['time'].values.astype('datetime64[W]') - merge_df['start'].values.astype('datetime64[W]'))
        weekly_event_count = merge_df.groupby(['enrollment_id','week_index']).event.count().unstack(fill_value=0)
        weekly_event_count = weekly_event_count.add_suffix("_count").rename(columns={'0 days 00:00:00_count': 'week_one_count', '7 days 00:00:00_count': 'week_two_count', 
        '14 days 00:00:00_count': 'week_three_count', '21 days 00:00:00_count': 'week_four_count', '28 days 00:00:00_count': 'week_five_count', '35 days 00:00:00_count': 'week_six_count'})
        weekly_event_count.reset_index(inplace=True)

        print("weekly_event_count features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the weekly_event_count dataframe: ", weekly_event_count.shape)
        return weekly_event_count
        
    def event_count(self):
        start_time = time.time()
        
        log_df = self.log.get_data()
        event_count_df = log_df.groupby(['enrollment_id']).event.value_counts().unstack(fill_value=0)
        event_count_df = event_count_df.add_suffix("_count")
        event_count_df.reset_index(inplace=True)
        
        print("event_count features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the event_count dataframe: ", event_count_df.shape)
        
        return event_count_df
    
    def get_features_all(self):
        event_count_df = self.event_count()
        # weekly_event_count = self.weekly_event_count()
        # features_df = pd.merge(event_count_df, weekly_event_count, how='inner', on='enrollment_id')
        return event_count_df
    
    def get_values_all(self):
        features_df = self.get_features_all()
        df_all = features_df.merge(self.truth.get_data(), left_on='enrollment_id', right_on='enrollment_id', how='inner')
        df_all_shuffled = df_all.sample(frac=1)
        
        X = df_all_shuffled.drop(columns=['label', 'enrollment_id']).values
        y = df_all_shuffled['label'].values
        
        print("Getting raw values for X, y done!")
        print("The shape of X: %s; shape of y: %s" % (X.shape, y.shape))
        return X, y

    def get_values_partial(self, ratio):
        features_df = self.get_features_all()
        df_all = features_df.merge(self.truth.get_data(), left_on='enrollment_id', right_on='enrollment_id', how='inner')
        df_1 = df_all[(df_all['label']==1)].sample(frac=ratio)
        df_partial = pd.concat([df_1, df_all[(df_all['label']==0)]])
        df_partial_shuffled = df_partial.sample(frac=1)

        X = df_partial_shuffled.drop(columns=['label', 'enrollment_id']).values
        y = df_partial_shuffled['label'].values

        print("Getting partial X, y done!")
        print("The shape of X: %s; shape of y: %s" % (X.shape, y.shape))
        y_ratio = np.count_nonzero(y)/len(y)
        print("The ratio of 1 in labels: ", "{:.2%}".format(y_ratio))

        return X, y