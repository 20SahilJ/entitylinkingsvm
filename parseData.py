#import statements
import numpy as np
import pandas as pd

#reads in x data from the wes tsv file
data = pd.read_csv('wes2015.tsv', sep = '\t', header = None, names = ['mentions', 'entities'])
#gets a dataframe of only the entities
entities = data['entities']
#removes the link part of every entity so that the entity ranker can properly analyze the entities
for i in range(0,len(entities)):
    entity=data.iloc[i,1]
    data.iloc[i,1] = entity.replace('http://dbpedia.org/resource/', '')
    entity = data.iloc[i, 1]
    data.iloc[i, 1] = entity.replace('http://de.dbpedia.org/resource/', '')
    entity = data.iloc[i,1]
    data.iloc[i,1] = entity.replace('http://dbpedia.org/', '')
#switch order of columns so that entities are in the first column and mentions are in the second column
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]
#gets a list of all the nonduplicate entities and their corresponding mentions. Note that this includes one occurance of
#  each entity that is duplicate as well.
nonduplicatelist = data['entities'].drop_duplicates(keep='first')
nonduplicates = data.loc[nonduplicatelist.index.tolist(),:]
#An array of all the duplicates. Note that one of the duplicate entities(along with its mentions) are in the
# nonduplicates dataframe
dups = data.loc[data['entities'].duplicated(keep='first'), :]
duplicates = dups
#calculates the max number of mentions a single entity can have. This is done by repeatedly finding duplicates(an array)
#  and adding 1 every time to a variable while the size of this duplicates array is not 0. (This duplicates array gets
# smaller because you consistently find the duplicates in that duplicates array, only eliminating one entity for each
# duplication each tme
extracols = 1
while len(duplicates)>0:
    duplicates = duplicates.loc[duplicates['entities'].duplicated(keep='first'),:]
    extracols = extracols + 1

#Creates a dataframe with the length of the data. Generation of random numbers is done simply to define length for
# amount of rows for which 'a' should be repeated, as this was the recommended way to get a column of repeated strings.
#This column of "a"s is then repeatedly added while the extracolumns count is not 0.
extracolsforuse = extracols
string = 'a'
numbers = np.random.randn(len(data))
tf = pd.DataFrame({'labels': string , 'numbers': numbers})
tf = tf['labels']
while extracolsforuse>0:
    nonduplicates[str(extracolsforuse)] = tf
    extracolsforuse = extracolsforuse - 1
#replaces all 'a's with the entity mentions for each entity that is duplicated. Nonduplicated entities are
# iterated in the first for loop, and their duplicate mentions(assuming one mention is already accounted for) are
# iterated in the inner for loop. For cases where there are multiple mentions for each entity, a while loop is used to
# check the first index does not have an 'a', or the space available to put in another mention for that entity.
for j in range(0,len(nonduplicatelist)):
    entity = nonduplicatelist.iloc[j]
    for i in range(0,len(dups)):
        currentity = dups.iloc[i, 0]
        if currentity==entity:
            extracolsforuse = 2
            while nonduplicates.iloc[j,extracolsforuse]!='a':
                extracolsforuse = extracolsforuse + 1
            nonduplicates.iloc[j,extracolsforuse] = dups.iloc[i,1]
#Originally creates y data by simply making a dataframe of size similar to the x data and full of zeros. The 'entity'
# column in the y data is then removed.
y = pd.DataFrame(0, index=np.arange(len(nonduplicates)), columns=nonduplicates.columns.values)
y = y.loc[:,y.columns!='entities']
#Edits both X(nonduplicates) data and y data. Edits x data by making sure every set of words(whether it be an entity or
# a mention) is seperated by spaces rather than '_'. i.e. Golden_State_Warriors would be changed to Golden State
# Warriors for appropriate evaluation by the entity ranker. Edits y data by replacing places where there is an actual
#mention to be a 1 instead of a 0 to indicate that the mention at that place correctly corresponds to the entity
# represented on that line in the X data
print('Got here')
for i in range(0, len(nonduplicates)):
    index = 0
    for j in range(0,extracols+2):
        curritem = nonduplicates.iloc[i,j]
        nonduplicates.iloc[i,j] = curritem.replace("_", " ")
        curritem = nonduplicates.iloc[i, j]
        nonduplicates.iloc[i, j] = curritem.rstrip()
        if curritem!='a' and j!=0:
            y.iloc[i,j-1] = 1
            index = index + 1
    #checking for duplicate mentions for each entity
    curr = nonduplicates.iloc[i, 1:index+1]
    currupdated = list(set(curr))
    currupdatedlen = len(currupdated)
    numAs = extracols+1-currupdatedlen
    finallist = [nonduplicates.iloc[i,0]]+ currupdated
    As = ['a']*numAs
    finallist = finallist+As
    nonduplicates.iloc[i] = finallist



#Use these lines to write final data to the files. nonduplicates represents the x data and y represents the y data
nonduplicates.to_csv('newdata.txt', sep = '|', header=None, index=False)
y.to_csv('newresults.txt', sep = '|', header=None, index=False)