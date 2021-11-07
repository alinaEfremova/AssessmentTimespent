# header for initial data
header = ['id',
          'name',
          'comp_level',
          'duration',
          'lang',
          'students_num',
          'TA_FTE',
          'TA_help',
          'cog_layer',
          '_U_experience',
          '_U_qualification',
          '_U_didac',
          'NGA_num',
          'GA_num',
          'grade_source',
          'max_assignment_type',
          'next_assignment_type',
          'depend_assignments',
          '_U_first_prep',
          '_U_first_type',
          '_U_first_time',
          'volume',
          'requirements',
          'feedback',
          'soft_use',
          'groups',
          '_U_why',
          '_U_what',
          '_U_how',
          '_U_which',
          '_U_transparency',
          '_U_scoring_type',
          '_U_duration',
          '_U_reuse',
          '_U_design',
          '_U_rubric',
          'prep_time',
          'assess_time',
          'grad_time',
          'feedback_time']

# dependant variables
dependant = ["prep_time", "assess_time", "grad_time", "feedback_time"]

# independent variables
independent = ['students_num',
               'TA_FTE',
               'TA_help',
               'NGA_num',
               'GA_num',
               'grade_source',
               'max_assignment_type',
               'next_assignment_type',
               'depend_assignments',
               'volume',
               'requirements',
               'feedback',
               'soft_use',
               'groups',
               'comp_level',
               'duration',
               'lang',
               'cog_layer']

# substitutions to group answers
substitution = {'max_assignment_type': {'Tests/exam with closed questions': "test",
                                        'Tests/exam with open questions': 'test',
                                        'Tests/exam with questions of both types': 'test',
                                        'Assignments with open questions implying detailed answer': 'open',
                                        'Short problem-solving assignments': 'open',
                                        'Project-like problem-solving assignments': 'open',
                                        'Presentations': 'oral',
                                        'Oral tests/exams': 'oral',
                                        'Other (from question 3)': 'open'},
                'next_assignment_type': {'Tests/exam with closed questions': "test",
                                         'Tests/exam with open questions': 'test',
                                         'Tests/exam with questions of both types': 'test',
                                         'Assignments with open questions implying detailed answer': 'open',
                                         'Short problem-solving assignments': 'open',
                                         'Project-like problem-solving assignments': 'open',
                                         'Presentations': 'oral',
                                         'Oral tests/exams': 'oral',
                                         'Other (from question 3)': 'open'},
                'comp_level': {"Beginning (e.g. begining Bachelor's courses)": "Beginning",
                               "Intermediate (e.g. begining Master's courses, specializing Bachelor's courses)": "Intermediate",
                               "Advanced (e.g. specializing Master's courses)": "Advanced"}}

# continuous variables
continuous = ["students_num", "TA", "NGA_num", "GA_num", "assignments_volume", "duration", "groups", "requirements",
              "feedback", "grade_source"]

# categorical variables with two answer options
t_categorical = ["lang", "depend_assignments", "soft_use"]

# categorical variables with three answer options
anova_categorical = ["max_assignment_type", "comp_level", "cog_layer"]


main_vars = ["students_num", "TA_FTE", "TA_help",
             "NGA_num", "GA_num", "grade_source",
             "max_assignment_type", "next_assignment_type", "depend_assignments"]
add_vars = ["requirements", "feedback", "soft_use", "groups", "assignments_volume"]
