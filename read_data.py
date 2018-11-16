import pandas as pd
import numpy as np

def get_time_dict():
	# doc
	# 2
	# 3
    rng = pd.date_range('2013-10-27', '2014-08-01')
    print('number of dates:', len(rng))
    time_dict = pd.Series(np.arange(len(rng)), index=rng)
    print(time_dict['2013/10/30'])
    return time_dict




class Enrollment():
    def __init__(self, filename):
        fin = open(filename)
        fin.next()

        self.enrollment_ids = []
        self.enrollment_info = {}
        self.user_info = {}
        self.user_enrollment_id = {}
        self.course_info = {}

        for line in fin:
            enrollment_id, username, course_id = line.strip().split(',')
            if enrollment_id == 'enrollment_id':        # ignore the first row
                continue
            self.enrollment_ids.append(enrollment_id)
            self.enrollment_info[enrollment_id] = [username, course_id]

            if username not in self.user_info:
                self.user_info[username] = [course_id]
                self.user_enrollment_id[username] = [enrollment_id]
            else:
                self.user_info[username].append(course_id)
                self.user_enrollment_id[username].append(enrollment_id)

            if course_id not in self.course_info:
                self.course_info[course_id] = [username]
            else:
                self.course_info[course_id].append(username)
        print("load enrollment info over!")
        print("number of courses:", len(self.course_info))
        print("number of enrollments:", len(self.enrollment_info))
        print("information of enrollment_id=1:", self.enrollment_info.get("1"))

time_dict = get_time_dict()
enrollment = Enrollment('data/train/enrollment_train.csv')

