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

    def event_count(self):
        start_time = time.time()
        
        log_df = self.log.get_data()
        event_count_df = log_df.groupby(['enrollment_id']).event.value_counts().unstack(fill_value=0)
        event_count_df = event_count_df.add_suffix("_count")
        event_count_df.reset_index(level=0, inplace=True)
        
        print("event_count features extracted! %f seconds taken" % (time.time()-start_time))
        print("Shape of the dataframe: ", event_count_df.shape)
        
        return event_count_df