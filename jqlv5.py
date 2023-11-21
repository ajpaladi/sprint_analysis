import requests
from getpass import getpass
import pandas as pd
import plotly.express as px
import pprint
import numpy as np
from dateutil.parser import parse

class JiraReport:

    #allsprints
    #allteams
    #PI4
    #global variables
    MAX_RESULTS = 1000
    base_url = "https://dronebase.atlassian.net"
    username = "andy.paladino@zeitview.com"
    api_token = 'ATATT3xFfGF0R05N27KKoU3vqpBZkeTheDTAn-qCTeYmDoaT_mEuhK9XMSQ_8esv_b-R0kLEvoDIz5lFxTPHeovFHwYNMkK-9VRVtJmmwKcQH0ZWJRaLsh1N7Zn9ugakp8rlzlWJ7wirJRBlDLUML_SwnBloPvAr1dwlQBuI6HKS2gvygSk42K4=A35EC1DF'
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    def __init__(self, sprint_name):
        self.sprint_name = sprint_name

    def jira_request(self, jql, start_at=0, max_results=None):
        if max_results is None:
            max_results = self.MAX_RESULTS

        url = f"{self.base_url}/rest/api/3/search"
        query = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results
        }
        response = requests.get(url, headers=self.headers, params=query, auth=(self.username, self.api_token))
        response.raise_for_status()  # raise exception if request failed
        return response.json()

    def jql_to_table(self, jql, max_results=None):
        if max_results is None:
            max_results = self.MAX_RESULTS

        ########################
        ##### JQL TO TABLE #####
        ########################

        start_at = 0
        total = 1  # initial placeholder value
        issues = []

        while start_at < total:
            response = self.jira_request(jql, start_at=start_at, max_results=max_results)
            issues.extend(response["issues"]) #all issues
            total = response["total"]
            start_at += max_results


        jql_dict = {'id':[], 'key':[], 'summary':[], 'status':[], 'status_category':[], 'assignee':[],
                    'created':[], 'creator':[], 'last_status_change':[], 'last_updated':[],
                    'issuetype':[], 'storypoints':[], 'priority':[], 'subtask':[], 'sprint_id':[], 'sprint_name':[],
                    'sprint_start_date':[], 'sprint_end_date':[], 'sprint_state':[],
                    'labels':[], 'project':[], 'projectkey':[], 'subtasks':[], 'description':[],
                    'watch_count':[], 'parentkey':[], 'parentname':[], 'parentstatus':[]}


        for i in issues:
            #pprint.pprint(i)
            #print('\n')
            id = i['id']
            key = i['key']
            summary = i['fields']['summary']
            status = i['fields']['status']['name']
            status_category = i['fields']['status']['statusCategory']['name']

            try:
                assignee = i['fields']['assignee']['displayName']
            except TypeError:
                assignee = None


            created = i['fields']['created']
            creator = i['fields']['creator']['displayName']
            last_status_change = i['fields']['statuscategorychangedate']
            last_updated = i['fields']['updated']
            issuetype = i['fields']['issuetype']['name']

            try:
                if issuetype == 'Sub-task':
                    storypoints = i['fields']['customfield_10407']
                elif issuetype == 'Epic':
                    continue
                else:
                    storypoints = i['fields']['customfield_10119']
            except TypeError:
                storypoints = 0

            priority = i['fields']['priority']['name']
            subtask = i['fields']['issuetype']['subtask'] # may not need this

            sprints =  i['fields']['customfield_10113']
            desired_sprint_prefix = self.sprint_name[:9]
            desired_sprint = [sprint for sprint in sprints if sprint['name'].startswith(desired_sprint_prefix)]

            for desired_sprint_info in desired_sprint:
                sprint_name = desired_sprint_info['name']
                sprint_id = desired_sprint_info['id']
                sprint_state = desired_sprint_info['state']
                sprint_start_date = desired_sprint_info.get('startDate')
                sprint_end_date = desired_sprint_info.get('endDate')

            labels = i['fields']['labels']
            project = i['fields']['project']['name']
            projectkey = i['fields']['project']['key']
            subtasks = i['fields']['subtasks']
            description = i['fields']['description']
            watch_count = i['fields']['watches']['watchCount']

            try:
                parentkey = i['fields']['parent']['key'] ### need to change to if parent exist, print, else parent = None
                parentname = i['fields']['parent']['fields']['summary']
                parentstatus = i['fields']['parent']['fields']['status']['name']
            except (KeyError, TypeError):
                parentkey = None
                parentname = None
                parentstatus = None

            jql_dict['id'].append(id)
            jql_dict['key'].append(key)
            jql_dict['summary'].append(summary)
            jql_dict['status'].append(status)
            jql_dict['status_category'].append(status_category)
            jql_dict['assignee'].append(assignee)
            jql_dict['created'].append(created)
            jql_dict['creator'].append(creator)
            jql_dict['last_status_change'].append(last_status_change)
            jql_dict['last_updated'].append(last_updated)
            jql_dict['issuetype'].append(issuetype)
            jql_dict['storypoints'].append(storypoints)
            jql_dict['priority'].append(priority)
            jql_dict['subtask'].append(subtask)
            jql_dict['sprint_name'].append(sprint_name)
            jql_dict['sprint_id'].append(sprint_id)
            jql_dict['sprint_start_date'].append(sprint_start_date) # was commented before
            jql_dict['sprint_end_date'].append(sprint_end_date)     # was commented before
            jql_dict['sprint_state'].append(sprint_state)
            jql_dict['labels'].append(labels)
            jql_dict['project'].append(project)
            jql_dict['projectkey'].append(projectkey)
            jql_dict['subtasks'].append(subtasks)
            jql_dict['description'].append(description)
            jql_dict['watch_count'].append(watch_count)
            jql_dict['parentkey'].append(parentkey)
            jql_dict['parentname'].append(parentname)
            jql_dict['parentstatus'].append(parentstatus)

        report = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in jql_dict.items() ]))
        report['created'] = pd.to_datetime(report['created'])
        report['last_status_change'] = pd.to_datetime(report['last_status_change'])
        report['last_updated'] = pd.to_datetime(report['last_updated'])
        report.to_csv('output/report_pre.csv')

        ######################################
        ### cycletime from Created to Done ###
        ######################################

        def calculate_cycle_time(row):
            cycle_time = row['last_status_change'] - row['created']
            #cycle_time = row['last_updated'] - row['created']
            cycle_time_days = cycle_time.total_seconds() / (24 * 60 * 60)
            return cycle_time_days

        report['cycle_time_from_created'] = report.apply(calculate_cycle_time, axis=1)
        report['cycle_time_from_created'] = report['cycle_time_from_created'].apply(lambda x: round(x, 2))

        report['assignee'].fillna('Not Assigned', inplace=True)
        report['storypoints'].fillna(0, inplace=True)
        report.to_csv('output/JQLreport.csv')

        #################
        ### changelog ###
        #################

        changelog = {'project':[], 'id':[], 'key':[], 'changed_timestamp':[], 'fieldtype':[], 'fromStatus':[], 'toStatus':[]}
        project = report['projectkey'].iloc[0]

        for id, key in zip(report['id'].unique(), report['key'].unique()):
            url = f"{self.base_url}/rest/agile/1.0/issue/{id}"
            query = {"expand": "changelog"}

            try:
                response = requests.get(url, headers=self.headers, params=query, auth=(self.username, self.api_token))
                response.raise_for_status()
                response = response.json()
            except Exception as e:
                print(f"Failed to fetch data for ID: {id} due to {str(e)}")
                continue

            for r in response['changelog']['histories']:
                #pprint.pprint(r)
                #print('\n')

                changed_timestamp = r['created']
                fieldtype = r['items'][0]['field']
                fromStatus = r['items'][0]['fromString']
                toStatus = r['items'][0]['toString']

                changelog['project'].append(project)
                changelog['id'].append(id)
                changelog['key'].append(key)
                changelog['changed_timestamp'].append(changed_timestamp)
                changelog['fieldtype'].append(fieldtype)
                changelog['fromStatus'].append(fromStatus)
                changelog['toStatus'].append(toStatus)

        changelog_df = pd.DataFrame(changelog)
        changelog_df = changelog_df[(changelog_df.fieldtype == "status")]
        changelog_df['fromStatus'] = changelog_df['fromStatus'].replace('In Progress', 'In Development')
        changelog_df['toStatus'] = changelog_df['toStatus'].replace('In Progress', 'In Development')
        issue_status_mapping = {
            'Blocked': 'Development', #when an issue is blocked, it's still technically in development
            'Staging': 'Deployment',
            'Ready for Prod': 'Deployment',
            'Ready for RC': 'Deployment',
            'UAT/DR': 'Deployment',
            'In Development': 'Development',
            'Code Review': 'Development',
            "Won't Do": 'N/A',
            'To Do': 'Pre-Development',
            'Backlog': 'Pre-Development',
            'Define and Scope': 'Pre-Development',
            'Stakeholder Review': 'Pre-Development',
            'Design - In Progress': 'Pre-Development',
            'Needs Functional Grooming': 'Pre-Development',
            'Architecture Planning': 'Pre-Development',
            'Done': 'Production',
            'DEPLOYED TO PRODUCTION': 'Production',
            'Test In Progress': 'Development', #changed test in progress to development
            'QA Holding': 'Development', #changed test in progress to development
            'READY FOR QA': 'Development' #changed test in progress to development
        }
        changelog_df['fromStatus_summarized'] = changelog_df['fromStatus'].map(issue_status_mapping)
        changelog_df['toStatus_summarized'] = changelog_df['toStatus'].map(issue_status_mapping)
        changelog_df['changed_timestamp'] = pd.to_datetime(changelog_df['changed_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)
        changelog_df.to_csv('output/changelog.csv', index=False)

        #####################################
        ### time started / time completed ###
        #####################################

        start_finish = {'id': [], 'key': [], 'started_time': [], 'completed_time': []}

        #every ticket will have a start
        ts = changelog_df[((changelog_df.fromStatus_summarized == "Pre-Development")) &
                          ((changelog_df.toStatus_summarized == "Development") |
                           (changelog_df.toStatus_summarized == "Deployment") |
                           (changelog_df.toStatus_summarized == "Production"))]

        tc = changelog_df[(changelog_df.toStatus == "DEPLOYED TO PRODUCTION") |
                          (changelog_df.toStatus == "Done") |
                          (changelog_df.toStatus == "Won't Do")|
                          (changelog_df.toStatus == "Cannot Reproduce")]


        ts = ts.sort_values('changed_timestamp', ascending=False)
        ts_latest = ts.groupby('key').first().reset_index()
        ts_latest.to_csv('output/ts_latest.csv')


        tc = tc.sort_values('changed_timestamp', ascending=False)
        tc_latest = tc.groupby('key').first().reset_index()

        #merge report with ts_lastest to get time started
        merged_ts = pd.merge(report, ts_latest[['id', 'changed_timestamp']], on='id', how='left')
        merged_ts = merged_ts.rename(columns={'changed_timestamp': 'time_started'})
        merged_ts.to_csv('output/merged_ts.csv')

        #after that merge, merge with tc_latest to get time completed
        merged_all = pd.merge(merged_ts, tc_latest[['id', 'changed_timestamp']], on='id', how='left')
        merged_all = merged_all.rename(columns={'changed_timestamp': 'time_completed'})
        merged_all['time_started'].fillna(pd.NaT, inplace=True)
        merged_all['time_completed'].fillna(pd.NaT, inplace=True)
        merged_all.to_csv('output/merged_all.csv')

        #merged_all.loc[merged_all['status_category'] != 'Done', 'time_completed'] = np.datetime64('NaT')

        def calculate_cycle_time(row):
            if pd.isnull(row['time_started']) or pd.isnull(row['time_completed']):
                return np.nan

            cycle_time = row['time_completed'] - row['time_started']
            cycle_time_days = cycle_time.total_seconds() / (24 * 60 * 60)
            return round(cycle_time_days, 2)

        merged_all['cycle_time_from_started'] = merged_all.apply(calculate_cycle_time, axis=1)
        merged_all['cycle_time_from_started'] = merged_all['cycle_time_from_started'].apply(lambda x: round(x, 2))
        merged_all.loc[merged_all['cycle_time_from_started'] < 0, 'cycle_time_from_started'] = np.datetime64('NaT')
        merged_all.to_csv('output/report.csv')

        return merged_all, changelog_df

    def run_report(self):
        projects = ['TOOLS', 'ENT', 'IDC', 'SOL', 'SA', 'SM', 'REN', 'AIML', 'DB']

        self.total_report = pd.DataFrame()
        self.total_changelog = pd.DataFrame()

        for p in projects:
            if p == 'TOOLS':
                if self.sprint_name == "2023PI4S1":
                    # Now, update sprint name for the second part of TOOLS project
                    sprint_for_project = f"{self.sprint_name} - {p}"
                    project_report, changelog = self.jql_to_table(
                        f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                    self.total_report = pd.concat([self.total_report, project_report])

                elif self.sprint_name == "2023PI4S2":
                    # Now, update sprint name for the second part of TOOLS project
                    sprint_for_project = f"{self.sprint_name} - {p} - TEAM 1"
                    project_report, changelog = self.jql_to_table(
                        f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                    self.total_report = pd.concat([self.total_report, project_report])

                    # Now, update sprint name for the second part of TOOLS project
                    sprint_for_project = f"{self.sprint_name} - {p} - TEAM 2"
                    project_report, changelog = self.jql_to_table(
                        f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                    self.total_report = pd.concat([self.total_report, project_report])
                else:
                    # Use different sprint names for TOOLS project
                    sprint_for_project = f"{self.sprint_name} - {p} 🚀"
                    project_report, changelog = self.jql_to_table(
                        f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                    self.total_report = pd.concat([self.total_report, project_report])

                    # Now, update sprint name for the second part of TOOLS project
                    sprint_for_project = f"{self.sprint_name} - {p} 🏆"
                    project_report, changelog = self.jql_to_table(
                        f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                    self.total_report = pd.concat([self.total_report, project_report])

            elif self.sprint_name == "2023PI4S2" and p == "SOL":
                # Exclude SOL for 2023PI4S2
                continue

            else:
                # For other projects, use the regular sprint name
                sprint_for_project = f"{self.sprint_name} - {p}"
                project_report, changelog = self.jql_to_table(
                    f"project = {p} AND sprint = '{sprint_for_project}' ORDER BY created DESC", 1000)
                self.total_report = pd.concat([self.total_report, project_report])

            self.total_changelog = pd.concat([self.total_changelog, changelog])
            self.total_report.to_csv('output/sprint_report.csv')  # JIRA OUTPUT
            self.total_report['sprint_end_date'] = pd.to_datetime(self.total_report['sprint_end_date'], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)
            self.total_report_completed = self.total_report[(self.total_report['status_category'] == 'Done') & (self.total_report['time_completed'] <= self.total_report['sprint_end_date'])]
            self.total_report_completed.to_csv('output/sprint_report_completed.csv')  # JIRA OUTPUT

        mapping_data = {
            'project_name': ['Internal Tools', 'Enterprise Capture', 'Insights-Data Capture', 'Solar Insights', 'Solar - Analysis', 'Solar - Moblie', 'Wind', 'AI/ML', 'Properties'],
            'project': ['TOOLS', 'ENT', 'IDC', 'SOL', 'SA', 'SM', 'REN', 'AIML', 'DB']
        }

        mapping_df = pd.DataFrame(mapping_data)
        self.total_changelog = self.total_changelog.merge(mapping_df, on='project', how='left')
        self.total_changelog = self.total_changelog.merge(self.total_report[['key', 'issuetype']], on='key', how='left')
        self.total_changelog.to_csv('output/changelog.csv')

        return self.total_report, self.total_report_completed, self.total_changelog

    def time_in_status(self):

        ########################
        #### Time in Status ####
        ########################

        changelog = self.total_changelog
        changelog = pd.merge(changelog, self.total_report, on ='key', how='left')
        changelog.to_csv('output/changelog_merge.csv')
        changelog_done = changelog[(changelog['status_category'] == "Done") & (changelog['time_completed'] <= changelog['sprint_end_date'])]        #filter for all completed issues potentially add a wont do here

        # Convert timestamp column to datetime objects
        changelog_done['sprint_start_date'] = pd.to_datetime(changelog_done['sprint_start_date'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc = True)


        changelog_done = changelog_done[(changelog_done['changed_timestamp'] >= changelog_done['sprint_start_date']) &
                                        (changelog_done['changed_timestamp'] <= changelog_done['sprint_end_date'])]

        changelog_done.to_csv('output/changelog_done.csv')

        # Dictionary to store user stories history
        results = []

        # Calculate the time spent in each status for each user story
        for index, row in changelog_done.iterrows():
            project_name = row['project_name']
            project = row['project_x']
            story_id = row['id_x']
            key = row['key']
            issuetype = row['issuetype_x']
            timestamp = row['changed_timestamp']
            status_from = row['fromStatus']
            status_to = row['toStatus']
            sprint_start_time = row['sprint_start_date']
            sprint_end_time = row['sprint_end_date']  # Assuming you have 'sprint_end_time' as a column in 'changelog_done'

            completed_statuses = ['Done', 'DEPLOYED TO PRODUCTION', "Won't Do"]

            if status_to in completed_statuses:
                time_spent_in_status = timestamp - sprint_start_time
            else:
                # Filter status changes within the sprint period
                sprint_status_changes = changelog_done.loc[(changelog_done['project_x'] == project)
                    & (changelog_done['id_x'] == story_id)
                    & (changelog_done['changed_timestamp'] >= sprint_start_time)
                    & (changelog_done['changed_timestamp'] <= sprint_end_time)
                ]
                sprint_status_changes = sprint_status_changes.sort_values(by='changed_timestamp', ascending=False)
                if not sprint_status_changes.empty:
                    next_status_time = sprint_status_changes['changed_timestamp'].iloc[0]
                else:
                    next_status_time = sprint_end_time  # If there is no next status change, set the next status time as the sprint end time

                time_spent_in_status = next_status_time - timestamp

            time_in_hours = time_spent_in_status.total_seconds() / 3600

            if time_in_hours < 0:
                print(f'Negative time found for project {project}, key {key}, story id {story_id}, from status {status_from}, timestamp {timestamp}, sprint start {sprint_start_time}, sprint end {sprint_end_time}, next status time {next_status_time}')

            results.append([project_name, project, key, issuetype, story_id, status_from, time_in_hours])

        self.time_in_status = pd.DataFrame(results, columns=['Project Name', 'Project', 'Key', 'IssueType', 'Id', 'Status', 'Time in Status (hours)'])

        status_mapping = {
            'Blocked': 'Blocked',
            'Staging': 'Deployment',
            'Ready for Prod': 'Deployment',
            'Ready for RC': 'Deployment',
            'UAT/DR': 'Deployment',
            'In Development': 'Development',
            'Code Review': 'Development',
            "Won't Do": 'N/A',
            'To Do': 'Pre-Development',
            'Backlog': 'Pre-Development',
            'Define and Scope': 'Pre-Development',
            'Done': 'Production',
            'DEPLOYED TO PRODUCTION': 'Production',
            'Test In Progress': 'QA',
            'QA Holding': 'QA',
            'READY FOR QA': 'QA'
        }

        # Create the new 'Summarized Status' column
        self.time_in_status['Summarized Status'] = self.time_in_status['Status'].map(status_mapping)
        self.time_in_status_mean = self.time_in_status.groupby(['Project', 'Status'])['Time in Status (hours)'].mean().reset_index()
        self.time_in_status_sum = self.time_in_status.groupby(['Project', 'Status'])['Time in Status (hours)'].sum().reset_index()

        #save to csv
        self.time_in_status.to_csv('output/time_in_status.csv', index = False) #JIRA OUTPUT
        self.time_in_status_mean.to_csv('output/time_in_status_mean.csv', index=False) #JIRA OUTPUT
        self.time_in_status_sum.to_csv('output/time_in_status_sum.csv', index=False) #JIRA OUTPUT

        return self.time_in_status, self.time_in_status_mean, self.time_in_status_sum

    def build_pivots(self):
        ############################
        #### Building the Pivots ###
        ############################

        ############################
        #### PIVOT TABLE ONE  ######
        ############################

        stories_completed = self.total_report_completed[((self.total_report_completed.status == 'DEPLOYED TO PRODUCTION') |
                                                    (self.total_report_completed.status == 'Done')) &
                                                   ((self.total_report_completed.issuetype == 'Bug') |
                                                    (self.total_report_completed.issuetype == 'Spike') |
                                                    (self.total_report_completed.issuetype == 'Story'))]



        stories_completed.to_csv('output/stories_completed_pivot.csv')
        stories_completed_count = stories_completed.groupby('project').agg({'id': 'count'}).reset_index()

        #grand total
        grand_total = stories_completed_count['id'].sum()
        self.stories_completed_count_initial = stories_completed_count.append({
            'project': 'Grand Total',
            'id': grand_total
        }, ignore_index=True)

        ####################################################
        #### PIVOT TABLE TWO: STORY POINTS COMPLETED  ######
        ####################################################

        story_points_completed = stories_completed.groupby('project').agg({'storypoints': 'sum'}).reset_index()
        grand_total_story_points = story_points_completed['storypoints'].sum()
        grand_total_row = {'project': 'grand total', 'storypoints': grand_total_story_points}
        self.story_points_completed_pivot = story_points_completed.append(grand_total_row, ignore_index=True)
        self.story_points_completed_pivot.to_csv('output/story_points_completed_pivot.csv')

        ############################
        #### PIVOT TABLE THREE #####
        ############################

        time_in_status_filter_one = self.time_in_status[((self.time_in_status['Summarized Status'] == 'Deployment') |
                                                (self.time_in_status['Summarized Status'] == 'Development') |
                                                (self.time_in_status['Summarized Status'] == 'QA')) &
                                               ((self.time_in_status.IssueType == 'Bug') |
                                                (self.time_in_status.IssueType == 'Sub-task') |
                                                (self.time_in_status.IssueType == 'Spike') |
                                                (self.time_in_status.IssueType == 'Story'))]
        time_in_status_filter_one.to_csv('output/tisf1.csv')

        time_in_status_pivot_one = time_in_status_filter_one.pivot_table(index='Project Name',
                                                                columns='Summarized Status',
                                                                values='Time in Status (hours)',
                                                                aggfunc='sum').fillna(0)


        total_hours_per_project_one = time_in_status_pivot_one.sum(axis=1)
        self.time_in_status_percentage_one = time_in_status_pivot_one.divide(total_hours_per_project_one, axis=0) * 100
        self.time_in_status_percentage_one = self.time_in_status_percentage_one.reset_index()

        #grand total
        total_hours_per_status = time_in_status_pivot_one.sum()
        grand_total_hours = total_hours_per_status.sum()
        grand_total_percentage = total_hours_per_status / grand_total_hours * 100
        grand_total_percentage['Project Name'] = 'Grand Total'
        self.time_in_status_percentage_one = self.time_in_status_percentage_one.append(grand_total_percentage, ignore_index=True)
        self.time_in_status_percentage_one.iloc[:, 1:] = self.time_in_status_percentage_one.iloc[:, 1:].applymap(lambda x: f'{x:.2f}%')


        ############################
        #### PIVOT TABLE FOUR #####
        ############################

        time_in_status_filter_two = self.time_in_status[(self.time_in_status['Status'] == 'Code Review') |
                                                   (self.time_in_status['Status'] == 'In Development') |
                                                   (self.time_in_status['Status'] == 'QA Holding') |
                                                   (self.time_in_status['Status'] == 'Ready for Prod') |
                                                   (self.time_in_status['Status'] == 'READY FOR QA') |
                                                   (self.time_in_status['Status'] == 'Ready for RC') |
                                                   (self.time_in_status['Status'] == 'Staging') |
                                                   (self.time_in_status['Status'] == 'Test In Progress') |
                                                   (self.time_in_status['Status'] == 'UAT/DR')]
        time_in_status_filter_two.to_csv('output/tisf2.csv')

        time_in_status_pivot_two = time_in_status_filter_two.pivot_table(index='Project Name',
                                                                columns='Status',
                                                                values='Time in Status (hours)',
                                                                aggfunc='sum').fillna(0)

        total_hours_per_project_two = time_in_status_pivot_two.sum(axis=1)
        self.time_in_status_percentage_two = time_in_status_pivot_two.divide(total_hours_per_project_two, axis=0) * 100
        self.time_in_status_percentage_two = self.time_in_status_percentage_two.reset_index()
        self.time_in_status_percentage_one.to_csv('output/time_in_status_percentage_one.csv')

        #grand total
        total_hours_per_status_two = time_in_status_pivot_two.sum()
        grand_total_hours_two = total_hours_per_status_two.sum()
        grand_total_percentage_two = total_hours_per_status_two / grand_total_hours_two * 100
        grand_total_percentage_two['Project Name'] = 'Grand Total'
        self.time_in_status_percentage_two = self.time_in_status_percentage_two.append(grand_total_percentage_two, ignore_index=True)
        self.time_in_status_percentage_two.iloc[:, 1:] = self.time_in_status_percentage_two.iloc[:, 1:].applymap(lambda x: f'{x:.2f}%')
        self.time_in_status_percentage_two.to_csv('output/time_in_status_percentage_two.csv')

        ############################
        #### PIVOT TABLE FIVE ######
        ############################

        breakdown_issues = self.total_report_completed[(self.total_report_completed.status == 'DEPLOYED TO PRODUCTION') |
                                                    (self.total_report_completed.status == 'Done')]

        self.breakdown_issues_pivot = breakdown_issues.pivot_table(index='project',
                                                    columns='issuetype',
                                                    values='id',
                                                    aggfunc='count').fillna(0)


        self.breakdown_issues_pivot['Grand Total'] = self.breakdown_issues_pivot.sum(axis=1)
        self.breakdown_issues_pivot.loc['Grand Total'] = self.breakdown_issues_pivot.sum()
        self.breakdown_issues_pivot = self.breakdown_issues_pivot.astype(int)
        total_bss = self.breakdown_issues_pivot[['Bug', 'Spike', 'Story']].sum(axis=1)
        self.breakdown_issues_pivot['% Bugs'] = (self.breakdown_issues_pivot['Bug'] / total_bss) * 100
        self.breakdown_issues_pivot['% Bugs'] = self.breakdown_issues_pivot['% Bugs'].round(2)  # Round to two decimal places
        self.breakdown_issues_pivot['% Bugs'] = self.breakdown_issues_pivot['% Bugs'].apply(lambda x: f'{x:.2f}%')
        self.breakdown_issues_pivot.to_csv('output/breakdown_issues_pivot.csv')


        ############################
        #### PIVOT TABLE SIX #######
        ############################

        cycle_time_breakdown = self.total_report_completed[(self.total_report_completed.issuetype == 'Bug') |
                                                      (self.total_report_completed.issuetype == 'Spike') |
                                                      (self.total_report_completed.issuetype == 'Story')]

        self.cycle_time_breakdown_pivot = cycle_time_breakdown.pivot_table(index='project',
                                                         values='cycle_time_from_started',
                                                         aggfunc=['mean', 'median']).fillna(0)


        self.cycle_time_breakdown_pivot.columns = [' '.join(col).strip() for col in self.cycle_time_breakdown_pivot.columns.values]
        self.cycle_time_breakdown_pivot.columns = ['AVERAGE of cycle_time_from_started', 'MEDIAN of cycle_time_from_started']

        # Calculate the grand total
        grand_total = cycle_time_breakdown['cycle_time_from_started'].agg(['mean', 'median'])
        grand_total.index = ['AVERAGE of cycle_time_from_started', 'MEDIAN of cycle_time_from_started']
        grand_total.name = 'Grand Total'
        self.cycle_time_breakdown_pivot = self.cycle_time_breakdown_pivot.append(grand_total)
        self.cycle_time_breakdown_pivot = self.cycle_time_breakdown_pivot.round(2)
        self.cycle_time_breakdown_pivot.to_csv('output/cycle_time_breakdown_pivot.csv')


        ##############################
        #### PIVOT TABLE SEVEN #######
        ##############################

        worked_v_completed = self.total_report[((self.total_report.status_category == 'In Progress') |
                                          (self.total_report.status_category == 'Done')) &
                                         ((self.total_report.issuetype == 'Bug') |
                                          (self.total_report.issuetype == 'Spike') |
                                          (self.total_report.issuetype == 'Story'))]


        worked_pivot = worked_v_completed.pivot_table(index='project',
                                                      values='id',
                                                      aggfunc='count').fillna(0)

        stories_completed_count.set_index('project', inplace=True)
        self.worked_v_completed_pivot = pd.concat([worked_pivot, stories_completed_count], axis=1)
        self.worked_v_completed_pivot.columns = ['Total Worked on', 'Completed']

        self.worked_v_completed_pivot['Percentage'] = (self.worked_v_completed_pivot['Completed'] / self.worked_v_completed_pivot['Total Worked on'] * 100).fillna(0)


        grand_total = self.worked_v_completed_pivot[['Total Worked on', 'Completed']].sum(numeric_only=True)
        grand_total['Percentage'] = (grand_total['Completed'] / grand_total['Total Worked on'] * 100)
        grand_total.name = 'Grand Total'
        self.worked_v_completed_pivot = self.worked_v_completed_pivot.append(grand_total)
        self.worked_v_completed_pivot['Percentage'] = self.worked_v_completed_pivot['Percentage'].round(2)
        self.worked_v_completed_pivot['Percentage'] = self.worked_v_completed_pivot['Percentage'].apply(lambda x: f'{x:.2f}%')

        self.breakdown_issues_pivot.reset_index(inplace=True)
        self.cycle_time_breakdown_pivot.reset_index(inplace=True)
        self.worked_v_completed_pivot.reset_index(inplace=True)
        self.worked_v_completed_pivot.to_csv('output/worked_v_completed_pivot.csv')

        return self.stories_completed_count_initial, self.time_in_status_percentage_one, self.time_in_status_percentage_two, self.breakdown_issues_pivot, self.cycle_time_breakdown_pivot, self.worked_v_completed_pivot

    def build_plotly_graphs(self):
        # fig1
        self.stories_completed_fig = px.bar(self.stories_completed_count_initial, x='project', y='id', title='Stories Completed')

        self.story_points_completed_fig = px.bar(self.story_points_completed_pivot, x='project', y='storypoints', title='Story Points Completed')

        # fig2
        self.time_in_status_fig_1 = px.bar(self.time_in_status_percentage_one, x='Project Name',
                                      y=['Deployment', 'Development', 'QA'],
                                      title='% of Time in Status: Deployment, Developement, QA')

        # fig3
        self.time_in_status_fig_2 = px.bar(self.time_in_status_percentage_two, x='Project Name',
                                      y=['Code Review', 'In Development', 'QA Holding', 'READY FOR QA', 'Ready for Prod',
                                         'Ready for RC', 'Staging', 'Test In Progress', 'UAT/DR'],
                                      title='% of Time in Status: Explicit Statuses')
        # fig4
        df_melted = self.breakdown_issues_pivot.melt(id_vars=['project'], value_vars=['Bug', 'Spike', 'Story', 'Sub-task'],
                                                var_name='issuetype', value_name='count')

        df_melted = df_melted[df_melted['count'] > 0]
        df_melted['label'] = df_melted['issuetype'] + ' (' + df_melted['count'].astype(str) + ')'

        self.df_melted_fig = px.sunburst(df_melted, path=['project', 'label'],
                                    values='count', title='Completed Issue Type Breakdown by Project & Issue Type',
                                    height=800)

        # fig5
        self.cycle_time_breakdown_fig = px.line(self.cycle_time_breakdown_pivot,
                                           x='project', y=['AVERAGE of cycle_time_from_started',
                                                           'MEDIAN of cycle_time_from_started'],
                                           title='Mean and Median Cycle Time Per Team')

        # fig6
        self.worked_v_completed_fig = px.bar(self.worked_v_completed_pivot, x='project', y=['Total Worked on', 'Completed'],
                                        title='Issues Worked vs. Completed, Per Team')

        return self.stories_completed_fig, self.time_in_status_fig_1, self.time_in_status_fig_2, self.df_melted_fig, self.cycle_time_breakdown_fig, self.worked_v_completed_fig

    def share_to_google(self):

        total_stories = self.total_report
        completed_stories = self.total_report_completed
        time_in_status_completed = self.time_in_status
        time_in_status_completed_mean = self.time_in_status_mean
        time_in_status_completed_sum = self.time_in_status_sum

        ####################################################
        #### ADDING RAW DATA TO GOOGLE SHEET ###############
        ###################################################

        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        from gspread_dataframe import set_with_dataframe

        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('googlekeys/jira-reporting-389003-cc1f132d9b38.json', scope)
        client = gspread.authorize(creds)

        new_sheet = client.create(f'Product Team - Sprint: {self.sprint_name} Report')
        sheet1 = new_sheet.get_worksheet(0)  # Get the first worksheet
        sheet1.update_title("Total Stories")  # Rename the first worksheet
        set_with_dataframe(sheet1, total_stories)

        sheet2 = new_sheet.add_worksheet(title = "Completed Stories", rows="300", cols="90")
        set_with_dataframe(sheet2, completed_stories)

        # Add a new worksheet to the existing google sheet and add data
        sheet3 = new_sheet.add_worksheet(title="Time In Status (Completed Issues)", rows="300", cols="90")
        set_with_dataframe(sheet3, time_in_status_completed)

        sheet4 = new_sheet.add_worksheet(title="Time In Status Mean (Completed Issues)", rows="300", cols="90")
        set_with_dataframe(sheet4, time_in_status_completed_mean)

        sheet5 = new_sheet.add_worksheet(title="Time In Status Sum (Completed Issues)", rows="300", cols="90")
        set_with_dataframe(sheet5, time_in_status_completed_sum)

        ############################################
        #### ADDING PIVOTS TO GOOGLE SHEET #########
        ############################################

        sheet6 = new_sheet.add_worksheet(title="Analysis", rows="1000", cols="1000")

        def batch_write_to_sheet(df, start_row, worksheet, title = None):
            if title:
                worksheet.update(f'A{start_row}', [[title]])  # Add the title in the first cell of the row
                start_row += 1  # Move to the next row
            values = [df.columns.tolist()] + df.values.tolist()
            end_row = start_row + len(values) - 1
            cell_range = f"A{start_row}:Z{end_row}"
            worksheet.update(cell_range, values)
            return end_row + 2

        start_row = 1

        start_row = batch_write_to_sheet(self.stories_completed_count_initial, start_row, sheet6, title ='Stories Completed')
        start_row = batch_write_to_sheet(self.story_points_completed_pivot, start_row, sheet6,title='Story Points Completed')
        start_row = batch_write_to_sheet(self.time_in_status_percentage_one, start_row, sheet6, title ='Time in Status (Summarized)')
        start_row = batch_write_to_sheet(self.time_in_status_percentage_two, start_row, sheet6, title ='Time in Status (Explicit)')
        start_row = batch_write_to_sheet(self.breakdown_issues_pivot, start_row, sheet6, title ='Completed Issue Breakdown & %Bugs')
        start_row = batch_write_to_sheet(self.cycle_time_breakdown_pivot, start_row, sheet6, title ='Cycle Time Breakdown')
        start_row = batch_write_to_sheet(self.worked_v_completed_pivot, start_row, sheet6, title ='Issues Worked vs. Completed')

        ############################################
        #### SHARE GOOGLE SHEET WITH USERS #########
        ############################################

        #user_emails = ['andy.paladino@zeitview.com']


        user_emails = ['andy.paladino@zeitview.com',
                       'ryan.anderson@zeitview.com',
                       'seth.beck@zeitview.com']

        for e in user_emails:
            new_sheet.share(e, perm_type='user', role='writer')

        message = f'Report has been successfully shared with {user_emails}'

        return message

if __name__ == "__main__":
    print('Running Jira Report...')
    sprint_names = ["2023PI5S1"]  # Add all the sprint names you want to process
    for sprint_name in sprint_names:
        pi = []
        jira = JiraReport(sprint_name=sprint_name)
        total_report, total_report_completed, total_changelog = jira.run_report()
        time_in_status, time_in_status_mean, time_in_status_sum = jira.time_in_status()
        pivots = jira.build_pivots()
        plots = jira.build_plotly_graphs()
        #pi.append(total_report_completed)
        message = jira.share_to_google()
        print(message)

    #pi_all = pd.concat(pi)
    #pi_all.to_csv('output/pi_all.csv')















#class JiraTeamReport:
#class PIReport:
#class PITeamReport:
