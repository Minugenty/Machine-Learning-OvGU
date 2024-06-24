import argparse
import csv
import math
import pandas as pd


def calc_entropy(df, entropyS, depth, target):

    storeIG = {}

    for feature in df.columns[0:-1]:
        unique_feature_value = df[feature].unique()
        feature_entropy = 0

        for feature_value in unique_feature_value:
            df1 = df[df[feature] == feature_value]
            value_entropy = 0
            #print(df1.head())
            for t in target:
                df2 = df1[df1[df1.columns[-1]] == t]
                #print(df2.head())
                if len(df2) > 0:
                    prop = -(len(df2) / len(df1)) * math.log(len(df2) / len(df1), len(target))
                else:
                    prop = 0
                value_entropy += prop
            feature_entropy += value_entropy * len(df1) / len(df)

        IG = entropyS - feature_entropy
        storeIG[feature] = IG
    #print(storeIG)

    selected_attribute = max(storeIG, key=storeIG.get)
    #print(selected_attribute)
    sel_attr_uv = df[selected_attribute].unique()
    new_df = pd.DataFrame()
    for feature_value in sel_attr_uv:
        df1 = df[df[selected_attribute] == feature_value]
        #print(df1.head())
        ve = 0

        for t in target:
            df2 = df1[df1[df1.columns[-1]] == t]
            #print(df2.head())
            if len(df2) == 0:
                prop = 0
            else:
                prop = -(len(df2) / len(df1)) * math.log(len(df2) / len(df1), len(target))
            ve += prop
        #print(feature_value, ve)
        new_df = df[df[selected_attribute] == feature_value].copy()
        new_df.drop(selected_attribute, axis=1, inplace=True)
        att = str(selected_attribute) + "=" + str(feature_value)
        if ve == 0:
            print(depth, att, ve, df.iloc[0, -1], sep=',')
        else:
            print(depth, att, ve, "no_leaf", sep=',')
            calc_entropy(new_df, entropyS, depth+1, target)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Your CSV data file")

    args = parser.parse_args()
    file = args.data

    with open(file) as File:
        data = csv.reader(File, delimiter=',')
        data = pd.DataFrame(data)

    # print(data.iloc[:, -1])
    features = []
    for i in data:
        features.append("att" + str(i))
    data.columns = features
    classes = data[data.columns[-1]].value_counts()
    entropyS = 0
    for clas in classes:
        p = clas / len(data)
        entropyS += (-p * math.log(p, len(classes)))
    target = data[data.columns[-1]].unique()
    if entropyS != 0:
        print("0", "root", entropyS, "no_leaf", sep=',')
        calc_entropy(data, entropyS, 1, target)
    else:
        print("0", "root", entropyS, data.iloc[0, -1], sep=',')
