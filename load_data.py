import pandas as pd
import numpy as np 

class Course_Date():
    """Course_Date class to load the data from date.csv
    
    Args: 
        filename (str): path of the file
    Attributes:
        get_data (df): get the pandas dataframe
        get_course_ids (list): get the list of courses id
        get_course_info (dict): get the dictionary of courses information
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.course_ids = []
        self.course_info = {}

        for _, row in self.data.iterrows():
            self.course_ids.append(row['course_id'])
            self.course_info[row['course_id']] = {}
            self.course_info[row['course_id']]['start'] = row['from']
            self.course_info[row['course_id']]['end'] = row['to']
            date_range = pd.date_range(start=row['from'], end=row['to'])
            self.course_info[row['course_id']]['date_range'] = date_range
            self.course_info[row['course_id']]['time_span'] = len(date_range)

        print('%s loaded! Number of courses: %i' % (filename, len(self.course_ids)))

    def get_data(self):
        return self.data
    
    def get_course_ids(self):
        return self.course_ids
    
    def get_course_info(self):
        return self.course_info


class Truth():
    """Truth class to load the data from truth_test/train.csv
    
    Args: 
        filename (str): path of the file
    Attributes:
        get_data (df): get the pandas dataframe
        get_labels (dict): get the dictionary of labels information with leys of enrollment_id
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename, names=['enrollment_id', 'label'])
        self.labels = {}
    
        for _, row in self.data.iterrows():
                self.labels[row['enrollment_id']] = row['label']
        
        print('%s loaded! Number of labels: %i' % (filename, len(self.labels)))

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


class Enrollment():
    """Enrollment class to load the data from enrollment_test/train.csv
    
    Args: 
        filename (str): path of the file
    Attributes:
        get_data (df): get the pandas dataframe
        get_enrollment_ids (list): get the list of enrollment ids
        get_enrollment_info (dict): get the dictionary of enrollment information
        get_course_info (dict): get the dictionary of the courses information
        get_user_info (dict): get the dictionary of the users information
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.enrollment_ids = []
        self.enrollment_info = {}
        self.user_info = {}
        self.course_info = {}

        for _, row in self.data.iterrows():
            self.enrollment_ids.append(row['enrollment_id'])
            self.enrollment_info[row['enrollment_id']] = {'username': row['username'], 'course_id': row['course_id']}

            if row['username'] not in self.user_info:
                self.user_info[row['username']] = {'course_id': [row['course_id']], 'enrollment_id': [row['enrollment_id']]}
            else:
                self.user_info[row['username']]['course_id'].append(row['course_id'])
                self.user_info[row['username']]['enrollment_id'].append(row['enrollment_id'])

            if row['course_id'] not in self.course_info:
                self.course_info[row['course_id']] = {'username': [row['username']]}
            else:
                self.course_info[row['course_id']]['username'].append(row['username'])
        
        for key in self.user_info:
            self.user_info[key]['course_number'] = len(self.user_info[key]['course_id'])
        
        for key in self.course_info:
            self.course_info[key]['user_number'] = len(self.course_info[key]['username'])

        print('%s loaded! Number of enrollments: %i' % (filename, len(self.enrollment_info)))

    def get_data(self):
        return self.data

    def get_enrollment_ids(self):
        return self.enrollment_ids
    
    def get_enrollment_info(self):
        return self.enrollment_info

    def get_course_info(self):
        return self.course_info

    def get_user_info(self):
        return self.user_info


class Course_Object():
    """Course_Object class to load the data from object.csv
    
    Args: 
        filename (str): path of the file
    Attributes:
        get_data (df): get the pandas dataframe
        get_module_ids (list): get the list of modules ids
        get_module_info (dict): get the dictionary of module information
        get_course_info (dict): get the dictionary of the courses information
        get_course_info_count (dict): get the dictionary of courses information with number of modules each course contains
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename).drop_duplicates(subset='module_id', keep='first')
        self.module_ids = []
        self.course_info = {}
        self.module_info = {}
        self.course_info_count = {}

        for _, row in self.data.iterrows():
            self.module_ids.append(row['module_id'])
            self.module_info[row['module_id']] = {'course_id': row['course_id'], 'category': row['category'], 
                                                'children': row['children'], 'start': row['start']}
            
            if row['course_id'] not in self.course_info:
                self.course_info[row['course_id']] = {row['module_id']: {'category': row['category'], 
                                                'children': row['children'], 'start': row['start']}}
            else:
                self.course_info[row['course_id']][row['module_id']] = {'category': row['category'], 
                                                'children': row['children'], 'start': row['start']}
        
        for key in self.course_info:
            self.course_info_count[key] = {'module_number': len(self.course_info[key])}

        print('%s loaded! Number of moduels: %i' % (filename, len(self.module_info)))

    def get_data(self):
        return self.data

    def get_module_ids(self):
        return self.module_ids

    def get_module_info(self):
        return self.module_info

    def get_course_info(self):
        return self.course_info

    def get_course_info_count(self):
        return self.course_info_count


class Log():
    """Log class to load the data from log_train/test.csv
    
    Args: 
        filename (str): path of the file
    Attributes:
        get_data (df): get the pandas dataframe
        get_data_by_id (df): get the dataframe by filtering with id list
        get_data_by_time (df): get the dataframe by filtering with start and end time
        get_data_by_id_time (df): get the dataframe by filtering with id list and time interval
    """
    def __init__(self, filename):
        self.data = pd.read_csv(filename).drop_duplicates()
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.enrollment_ids = self.data['enrollment_id'].unique()

        print('%s loaded! Size of log data: %i' % (filename, len(self.data['enrollment_id'])))
    
    def get_data(self):
        return self.data

    def get_enrollment_ids(self):
        return self.enrollment_ids

    def get_data_by_id(self, enrollment_ids):
        return self.data.loc[self.data['enrollment_id'].isin(enrollment_ids)]
    
    def get_data_by_time(self, start, end):
        mask = (self.data['time'] >= start) & (self.data['time'] <= end)
        return self.data.loc[mask]
    
    def get_data_by_id_time(self, ids, start, end):
        data = self.data.loc[self.data['enrollment_id'].isin(ids)]
        mask = (data['time'] >= start) & (data['time'] <= end)
        return data.loc[mask]
    