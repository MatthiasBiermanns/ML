import os
import pandas as pd
df_raw = pd.read_csv("raw.csv")
df_postprocessed = pd.read_csv("postprocessed.csv")
df_subtask = pd.read_csv("subtask_index.csv")
df_seq = pd.read_csv("seq_info.csv")


# Find all Files which will be deleted
def getPanoList():
    rawRows = df_raw.values
    panoPics = []
    for row in rawRows:
        if row[6] == True:
            panoPics.append(row[1])
    return panoPics


def getNightList():
    postprocessedRows = df_postprocessed.values
    nightPics = []
    for row in postprocessedRows:
        if row[4] == True:
            nightPics.append(row[1])
    return nightPics


def getW2SList():
    subtaskRows = df_subtask.values
    w2sPics = []
    for row in subtaskRows:
        if row[2] == True:
            w2sPics.append(row[1])
    return w2sPics


def getRemoveList():
    removeList = getPanoList() + getNightList() + getW2SList()
    return removeList


# Deletion of Files
def removeFiles(basePath: str, deletionList: list):
    for deletionObject in deletionList:
        filename = basePath + deletionObject + ".jpg"
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(filename + " not found")

# Remove keys from csv-Files


def removeRowsInCSV(removeList: list):
    df_r = pd.read_csv("raw.csv")
    df_p = pd.read_csv("postprocessed.csv")
    df_su = pd.read_csv("subtask_index.csv")
    df_se = pd.read_csv("seq_info.csv")
    for key in removeList:
        df_r = df_r[df_r.key != key]
        df_p = df_p[df_p.key != key]
        df_se = df_se[df_se.key != key]
        df_su = df_su[df_su.key != key]
    df_r.to_csv("raw.csv")
    df_p.to_csv("postprocessed.csv")
    df_se.to_csv("seq_info.csv")
    df_su.to_csv("subtask_index.csv")

# Main Runner


def main():
    removeList = getRemoveList()
    print(">>>>List containing " + str(len(removeList)))
    print(">>>>Starting to remove Files")
    removeFiles("./images/", removeList)
    print(">>>Starting to remove Rows")
    removeRowsInCSV(removeList)


main()
