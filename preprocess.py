import pandas as pd
import numpy as np 
from scipy.special import boxcox1p
import time
from datetime import datetime
from load_data import Course_Date
from load_data import Course_Object
from load_data import Log
from load_data import Enrollment
from load_data import Truth


class Preprocessor():
    def __init__(self, data_type):
        self.data_type = data_type
        self.course_date = Course_Date('data/date.csv')
        self.course_obj = Course_Object('data/object.csv')
        self.log = Log('data/%s/log_%s.csv' % (data_type, data_type))
        self.enroll = Enrollment('data/%s/enrollment_%s.csv' % (data_type, data_type))
        self.truth = Truth('data/%s/truth_%s.csv' % (data_type, data_type))

    def get_truth(self):
        return self.truth

    def problem_video_ratio(self):
        start_time = time.time()

        course_obj = self.course_obj.get_data()
        log = self.log.get_data()
        course_category = course_obj.groupby('course_id').category.value_counts().unstack(fill_value=0)
        course_category = course_category.filter(items=['problem', 'video'])
        course_category.reset_index(inplace=True)
        
        course_obj = course_category.merge(course_obj, on='course_id', how='inner')
        log_course_df = log.merge(course_obj, left_on='object', right_on='module_id', how='inner')
        log_course_df.drop_duplicates(subset=['enrollment_id', 'event', 'object'], inplace=True)
        problem_video_df = log_course_df.loc[log_course_df['event'].isin(['problem', 'video'])]

        event_count = problem_video_df.groupby('enrollment_id').event.value_counts().unstack(fill_value=0)
        event_count.reset_index(inplace=True)
        problem_video_df = problem_video_df.merge(event_count, on='enrollment_id',how='inner', suffixes=('_total', '_count'))
        problem_video_df = problem_video_df.filter(items=['enrollment_id', 'problem_total', 'video_total', 'problem_count','video_count']).drop_duplicates()
        problem_video_df = problem_video_df.assign(problem_ratio = problem_video_df.problem_count/problem_video_df.problem_total).assign(video_ratio = problem_video_df.video_count/problem_video_df.video_total)
        problem_video_df = problem_video_df.filter(items=['enrollment_id', 'problem_ratio','video_ratio'])
        problem_video_df = problem_video_df.merge(self.enroll.get_data().filter(['enrollment_id']), on='enrollment_id', how='right')
        problem_video_df.fillna(0, inplace=True)
        
        print("problem_video_ratio features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the problem_video dataframe: ", problem_video_df.shape)

        return problem_video_df

    def session_count(self):
        start_time = time.time()
        enroll_course_df = self.enroll.get_data().merge(self.course_date.get_data(), on='course_id', how='inner')
        merge_df = self.log.get_data().merge(enroll_course_df, on = 'enrollment_id', how='inner')

        merge_df = merge_df.filter(items=['enrollment_id', 'time', 'event'])
        merge_df = merge_df.assign(time_span = merge_df['time'].values.astype('datetime64[h]') - merge_df['time'].shift(1).values.astype('datetime64[h]'))
        session_series = merge_df.groupby(['enrollment_id']).apply(lambda x: (x['time_span'] != np.timedelta64(0,'ns')).sum())
        session_count_df = session_series.to_frame().reset_index().rename(columns={0:'session_count'})

        print("session_count features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the session_count dataframe: ", session_count_df.shape)

        return session_count_df

    def weekly_session_count(self):
        start_time = time.time()
        enroll_course_df = self.enroll.get_data().merge(self.course_date.get_data(), on='course_id', how='inner')
        merge_df = self.log.get_data().merge(enroll_course_df, on = 'enrollment_id', how='inner')

        merge_df = merge_df.filter(items=['enrollment_id', 'time', 'event', 'from'])
        merge_df= merge_df.assign(week_index = merge_df['time'].values.astype('datetime64[W]') - merge_df['from'].values.astype('datetime64[W]'))
        merge_df = merge_df.assign(time_span = merge_df['time'].values.astype('datetime64[h]') - merge_df['time'].shift(1).values.astype('datetime64[h]'))
        weekly_session_count = merge_df.groupby(['enrollment_id', 'week_index']).apply(lambda x: (x['time_span'] != np.timedelta64(0,'ns')).sum())
        weekly_session_count_df = weekly_session_count.to_frame()
        weekly_session_count_df = weekly_session_count_df.unstack(fill_value=0).sum(level=1,axis=1)
        weekly_session_count_df = weekly_session_count_df.add_suffix("_count").rename(columns={'0 days 00:00:00_count': 'week_one_session', '7 days 00:00:00_count': 'week_two_session', 
        '14 days 00:00:00_count': 'week_three_session', '21 days 00:00:00_count': 'week_four_session', '28 days 00:00:00_count': 'week_five_session', '35 days 00:00:00_count': 'week_six_session'})

        print("weekly_session_count features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the weekly_session_count dataframe: ", weekly_session_count_df.shape)

        return weekly_session_count_df


    def weekly_event_count(self):
        start_time = time.time()
        enroll_course_df = self.enroll.get_data().merge(self.course_date.get_data(), on='course_id', how='inner')
        merge_df = self.log.get_data().merge(enroll_course_df, on = 'enrollment_id', how='inner')

        merge_df = merge_df.filter(items=['enrollment_id', 'time', 'event', 'from'])
        merge_df= merge_df.assign(week_index = merge_df['time'].values.astype('datetime64[W]') - merge_df['from'].values.astype('datetime64[W]'))
        weekly_event_count = merge_df.groupby(['enrollment_id','week_index']).event.count().unstack(fill_value=0)
        weekly_event_count = weekly_event_count.add_suffix("_count").rename(columns={'0 days 00:00:00_count': 'week_one_event', '7 days 00:00:00_count': 'week_two_event', 
        '14 days 00:00:00_count': 'week_three_event', '21 days 00:00:00_count': 'week_four_event', '28 days 00:00:00_count': 'week_five_event', '35 days 00:00:00_count': 'week_six_event'})
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
        start_time = time.time()

        event_count_df = self.event_count()
        # weekly_event_count = self.weekly_event_count()
        session_count_df = self.weekly_session_count()
        problem_video_df = self.problem_video_ratio()
        event_session_count_df = pd.merge(event_count_df, session_count_df, how='inner', on='enrollment_id')
        features_df = pd.merge(event_session_count_df, problem_video_df, how='inner', on='enrollment_id')
        
        df_all = features_df.merge(self.truth.get_data(), left_on='enrollment_id', right_on='enrollment_id', how='inner')
        
        print("All features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the whole features dataframe: ", features_df.shape)

        return df_all
    
    def train_preprocessing(self):
        start_time = time.time()

        df = self.get_features_all()
'''        num_feature = list(df.columns[1:-1])

        for value in num_feature:
            freq_set = df[value].value_counts()
            max_1 = freq_set.max()
            max_2 = 0
            for num in freq_set:
                if num == max_1:
                    continue
                max_2 = num
                break
            ratio = max_2/float(max_1)
            if ratio < 0.05:
                df = df.drop(value,1)      
'''
        num_feature = list(df.columns[1:-1])
        for item in num_feature:
            df[item]= boxcox1p(df[item], 0.25)

        print("Getting train data preprocessing done! %f seconds taken" % (time.time()-start_time))
        return df
    
    def test_preprocessing(self):
        start_time = time.time()

        df = self.get_features_all()
        '''num_feature = list(df.columns[1:-1])

      
        df = df.drop('problem_count',1)
        df = df.drop('week_six_session',1)
        df = df.drop('problem_ratio',1)
'''
        num_feature = list(df.columns[1:-1])
        for item in num_feature:
            df[item]= boxcox1p(df[item], 0.25)

        print("Getting test data preprocessing done! %f seconds taken" % (time.time()-start_time))
        return df
    
    
    def get_values_all(self):
        if self.data_type == 'train':
            df_all = self.train_preprocessing()
        if self.data_type == 'test':
            df_all = self.test_preprocessing()
        
        df_all_shuffled = df_all.sample(frac=1)
        
        X = df_all_shuffled.drop(columns=['label', 'enrollment_id']).values
        y = df_all_shuffled['label'].values
        
        print("Getting raw values for X, y done!")
        print("The shape of X: %s; shape of y: %s" % (X.shape, y.shape))
        return X, y

    def get_values_partial(self, ratio):
        features_df = self.data_preprocessing()
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