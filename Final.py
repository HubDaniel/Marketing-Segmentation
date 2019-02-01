#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:21:43 2018
"""

import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import xlsxwriter
import warnings
warnings.filterwarnings("ignore")



def read_data():
    # inpout file name
    filename = input("Please Enter Filename with suffix: e.x. Toys Brand Website Data.xlsx\n")

    # inpout sheet position
    if '.xlsx' in filename:
        sheet_position = input("Please Enter Worksheet Position (starting with 1):\n")
        sheet_position = int(sheet_position) - 1

    # input header position
    header_position = input("Please Enter the Row Number of First Non-Header Entry in the File:\n(e.x. if it is the second row, enter 2)\n")
    header_position =int(header_position) - 2

    # read data
    path = os.path.join(os.path.dirname(__file__), filename)
    if '.csv' in filename:
         data = pd.read_csv(path, header=None)
    elif '.xlsx' in filename:
         data = pd.read_excel(path, sheet_name = sheet_position, header=None)
    else:
        print('file suffix is neither .csv or .xlsx')

    # combine column names
    if header_position > 0 :
        for i in range(0,data.shape[1]):
            if(pd.isnull(data.iloc[0,i])==True):
                data.iloc[0,i] = data.iloc[0,i-1]

        for i in range(0,data.shape[1]):
            if(pd.isnull(data.iloc[1,i])==True):
                data.iloc[1,i] = data.iloc[1,i-1]

        column_names = data.iloc[header_position,:] + "@@@@@" +  data.iloc[header_position-1,:]
        data.columns = list(column_names)

        data = data.iloc[header_position+1:,:]
    else:
         if '.csv' in filename:
             data = pd.read_csv(path, header=0)
         elif '.xlsx' in filename:
             data = pd.read_excel(path, sheet_name = sheet_position, header=0)
         else:
             print('file suffix is neither .csv or .xlsx')

    return data





def find_demo_columns(df, df_demo, not_demo):
    df_list = []
    for i in df.columns.tolist():
        for j in df_demo:
            if (j.upper() in i.upper()) &  (any([x.upper() in i.upper() for x in not_demo]) is False):
                    df_list.append(i)
    return(set(df_list))





def find_demo_data(data, demo_col):
    data_without_demo = data[[x for x in data.columns.tolist() if x not in demo_col]].copy()
    demo_data = data[[x for x in data.columns.tolist() if x in demo_col]].copy()

    return demo_data, data_without_demo





def demo_data_cleaning(demo_data):
    # null columns
    demo_data.fillna(-1, inplace=True)

    # categorical columns
    categorical_columns2 = list(demo_data.columns[demo_data.apply(lambda x: len(set(x)) < 15, axis =0)])

    # numeric columns
    numeric_columns2 = list(demo_data.drop(categorical_columns2, axis=1).select_dtypes(include=[np.number]).columns)

    # final demo_data
    ## unique columns names
    demo_data_cleaned = demo_data[list(set(numeric_columns2))+list(set(categorical_columns2))].copy()

    return demo_data_cleaned





def demo_data_engineering(demo_data):
    # categorical columns: dummy
    data_dummied2 = pd.get_dummies(demo_data)

    return data_dummied2





def data_without_demo_cleaning(data_without_demo):
    # null columns
    data_without_demo.fillna(-1, inplace=True)   # have warnings

    # categorical columns
    categorical_columns = list(data_without_demo.columns[data_without_demo.apply(lambda x: len(set(x)) < 15, axis =0)])

    # numeric columns
    numeric_columns = list(data_without_demo.drop(categorical_columns, axis=1).select_dtypes(include=[np.number]).columns)

    # datetime columns
    matching = [s for s in list(data_without_demo.columns) if "Date" in list(data_without_demo.columns)]
    datetime_columns = list(data_without_demo.select_dtypes(include=['datetime64']).columns)
    datetime_columns.extend(matching)

    # final data_without_demo
    ## unique columns names
    data_without_demo_cleaned = data_without_demo[list(set(numeric_columns))+list(set(categorical_columns)) + datetime_columns].copy()

    return data_without_demo_cleaned, categorical_columns, datetime_columns, numeric_columns





def data_without_demo_engineering(data_without_demo_cleaned, categorical_columns, datetime_columns):
    # categorical columns: dummy
    data_dummied = pd.get_dummies(data_without_demo_cleaned, columns = categorical_columns)

    # datetime columns: numeric
    for col in datetime_columns:
        data_dummied[col] = pd.to_numeric(data_dummied[col])

    # standardized all variables
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_without_demo_scaled = pd.DataFrame(scaler.fit_transform(data_dummied), columns = (data_dummied.columns))

    return data_dummied, data_without_demo_scaled





def run_kmeans(data_without_demo_scaled):
    ssd = []
    for i in range(2,10):
        km = KMeans(n_clusters=i, n_jobs=-1)
        km.fit(data_without_demo_scaled)
        ssd.append(km.inertia_)
    #plt.plot(range(2,10), ssd, '-o')
    #plt.show()

    return ssd





def elbow_method(ssd):
    values=list(ssd)
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    print ("Best number of clusters =",idxOfBestPoint + 2)
    return idxOfBestPoint





def get_best_k(idxOfBestPoint):
    best_k = idxOfBestPoint + 2

    km = KMeans(n_clusters=best_k, n_jobs=-1)
    km.fit(data_without_demo_scaled)

    # add label on original dataset (only after combining column names and fill na)
    df = pd.concat([data_without_demo, demo_data], axis=1)
    df['best_cluster_' + str(best_k)] = km.labels_

    # save output
    df.to_csv(os.path.join(os.path.dirname(__file__), 'original_data_with_labels.csv'), index=False)

    return df, best_k





def get_possible_ks(df, best_k):
    possible_k = [(best_k - 7) * ((best_k - 7) > 1),
                  (best_k - 6) * ((best_k - 6) > 1),
                  (best_k - 5) * ((best_k - 5) > 1),
                  (best_k - 4) * ((best_k - 4) > 1),
                  (best_k - 3) * ((best_k - 3) > 1),
                  (best_k - 2) * ((best_k - 1) > 2),
                  (best_k - 1) * ((best_k - 1) > 1),
                  best_k + 1,
                  best_k + 2,
                  best_k + 3,
                  best_k + 4,
                  best_k + 5]

    possible_k = [x for x in list(possible_k) if x!=0]
    possible_k

    for i in possible_k:
        km = KMeans(n_clusters=i, n_jobs=-1)
        km.fit(data_without_demo_scaled)
        df['cluster_'+str(i)] = km.labels_

    return df, possible_k





def cluster_summary(data_dummied, possible_k, best_k):
    data_dummied['best_cluster_' + str(best_k)] = df['best_cluster_' + str(best_k)]

    for counter,i in enumerate(possible_k):
        data_dummied["cluster_" + str(i)] = df['cluster_' + str(i)]
        if (counter+1) == len(possible_k):
            break

    df_all = pd.concat([data_dummied2, data_dummied], axis=1)

    workbook = xlsxwriter.Workbook(os.path.join(os.path.dirname(__file__),'ClusterSummary.xlsx'),  {'nan_inf_to_errors': True})
    cell_format = workbook.add_format({'bold': True})
    right_align = workbook.add_format({'align': 'right'})

    for counter, i in enumerate(np.append(best_k,possible_k).tolist()):

        if counter == 0:
            worksheet = workbook.add_worksheet(name='summary_best_' + str(i))
            summary = df_all.groupby('best_cluster_' + str(best_k)).agg('mean')
            summary.reset_index(inplace=True)
        else:
            worksheet = workbook.add_worksheet(name='cluster_'+ str(i) +'summary')
            summary = df_all.groupby('cluster_' + str(i)).agg('mean')
            summary.reset_index(inplace=True)


        row = 0
        col = 0
        for column in summary.columns:
            worksheet.write(row, col, column, cell_format)
            col += 1
        row = 1
        col = 0
        for i in range(len(summary)):
            series = summary.iloc[i,:].values
            for item in series:
                worksheet.write(row, col, item)
                col += 1
            row +=1
            col = 0
        for j in range(1,len(series)):
            worksheet.conditional_format(1,j,row, j, {'type': '2_color_scale', 'min_color': '#FFFFFF', 'max_color': '#008000'})
        worksheet.set_column(0, len(series), width=25)

    workbook.close()


if __name__ == '__main__' :
    df_demo = ['age', 'gender', 'state', 'child', 'income', 'living', 'industry', 'ethnicity', 'how old are you', 'To begin, we have some questions about you! Are you...', 'born', 'pregnant', "Education", 'Profession']
    not_demo = ['Age-by-Age', 'How does your [child age/gender] typically inform', 'Ask my child', 'In a typical month', 'page', 'When looking for toy related information', 'interesting to my child', 'statement', 'and images','average', 'agent', 'coverage']

    ### read data
    data = read_data()

    ### data cleaning and data engineering
    demo_columns = find_demo_columns(data, df_demo, not_demo)
    demo_data, data_without_demo = find_demo_data(data, demo_columns)
    demo_data_cleaned = demo_data_cleaning(demo_data)
    data_dummied2 = demo_data_engineering(demo_data)
    data_without_demo_cleaned, categorical_columns, datetime_columns, _ = data_without_demo_cleaning(data_without_demo)
    data_dummied, data_without_demo_scaled = data_without_demo_engineering(data_without_demo_cleaned, categorical_columns, datetime_columns)

    ### kmeans clustering
    ssd = run_kmeans(data_without_demo_scaled)
    idxOfBestPoint = elbow_method(ssd)
    df, K = get_best_k(idxOfBestPoint)
    df, Ks = get_possible_ks(df, K)
    cluster_summary(data_dummied, Ks, K)
