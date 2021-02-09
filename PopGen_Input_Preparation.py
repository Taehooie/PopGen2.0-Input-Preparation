
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import csv
import os


# In[2]:

class Marginals():
    
    def __init__(self, data1, data2, data3, data4, data5, data6):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5
        self.data6 = data6
        
    def generate_inputs(self, countyName, average7more):
        self.countyName = countyName
        
        HH_size = data1[data1['GEO.display-label'].str.contains(countyName)]
        HH_income = data2[data2['GEO.display-label'].str.contains(countyName)]
        marginal = data3[data3['COUNTY_NAM'].str.contains(countyName)]
        #marginal = marginal[marginal.TOTPOP12 != 0]
        marginal = marginal.sort_values(by='bpm2012TAZ')
        TAZ = pd.DataFrame({'TAZ_ID':marginal['bpm2012TAZ'], 'HHNUM12':marginal['HHNUM12'],
                             'HHSIZE12':marginal['HHSIZE12'], 'HHINCX12':marginal['HHINCX12']})
        TAZ = TAZ[['TAZ_ID','HHNUM12','HHSIZE12','HHINCX12']]
        
        Income_range = pd.concat([data4['HINCP'],data5['HINCP'],data6['HINCP']])
        Income_range = Income_range.dropna()
        
        Average7more = average7more
        
        return HH_size, HH_income, TAZ, Income_range, Average7more
    
    def Hsize_Distribution(self, countyName, average7more):
        
        # To ignore invalid calculation ( zero divided by zero)
        np.seterr(divide='ignore', invalid='ignore')
        
        # TODO: Calculating average household size 7 or more
        Avg_7more = self.generate_inputs(countyName, average7more)[4]

        
        df1 = self.generate_inputs(countyName, average7more)[0] 
        df2 = self.generate_inputs(countyName, average7more)[2] 
        df3 = self.generate_inputs(countyName, average7more)[1]
        data4 = self.generate_inputs(countyName, average7more)[3]
        
        householdnum = 7
    
        # This is subsetting work
        FamTable = df1[['HD01_VD10','HD01_VD03','HD01_VD11','HD01_VD04','HD01_VD12','HD01_VD05',
                        'HD01_VD13','HD01_VD06','HD01_VD14','HD01_VD07','HD01_VD15','HD01_VD08','HD01_VD16']]
    
        IncomeTable = df3[['HD01_VD02','HD01_VD03','HD01_VD04','HD01_VD05','HD01_VD06',
                           'HD01_VD07','HD01_VD08','HD01_VD09','HD01_VD10','HD01_VD11','HD01_VD12',
                           'HD01_VD13','HD01_VD14','HD01_VD15','HD01_VD16', 'HD01_VD17']] 
    
        
        # Processing dataframe to convert it to numpy.array
        datatable=FamTable.drop(FamTable.index[0]).values
        x=np.array(datatable)
        datatable=x.astype(np.float)
    
        datatable2=IncomeTable.drop(IncomeTable.index[0]).values
        x1=np.array(datatable2)
        datatable2=x1.astype(np.float)
    
        TotalIncome = data4
        x2 = np.array(TotalIncome)
        TotalIncome = x2.astype(np.float)
    
        mean1 = np.mean(TotalIncome[np.where(TotalIncome<10000)])
        mean2 = np.mean(TotalIncome[np.where((TotalIncome>=10000) & (TotalIncome<15000))])
        mean3 = np.mean(TotalIncome[np.where((TotalIncome>=15000) & (TotalIncome<20000))])
        mean4 = np.mean(TotalIncome[np.where((TotalIncome>=20000) & (TotalIncome<25000))])
        mean5 = np.mean(TotalIncome[np.where((TotalIncome>=25000) & (TotalIncome<30000))])
        mean6 = np.mean(TotalIncome[np.where((TotalIncome>=30000) & (TotalIncome<35000))])
        mean7 = np.mean(TotalIncome[np.where((TotalIncome>=35000) & (TotalIncome<40000))])
        mean8 = np.mean(TotalIncome[np.where((TotalIncome>=40000) & (TotalIncome<45000))])
        mean9 = np.mean(TotalIncome[np.where((TotalIncome>=45000) & (TotalIncome<50000))])
        mean10 = np.mean(TotalIncome[np.where((TotalIncome>=50000) & (TotalIncome<60000))])
        mean11 = np.mean(TotalIncome[np.where((TotalIncome>=60000) & (TotalIncome<75000))])
        mean12 = np.mean(TotalIncome[np.where((TotalIncome>=75000) & (TotalIncome<100000))])
        mean13 = np.mean(TotalIncome[np.where((TotalIncome>=100000) & (TotalIncome<125000))])
        mean14 = np.mean(TotalIncome[np.where((TotalIncome>=125000) & (TotalIncome<150000))])
        mean15 = np.mean(TotalIncome[np.where((TotalIncome>=150000) & (TotalIncome<200000))])
        mean16 = np.mean(TotalIncome[np.where((TotalIncome>=200000))])
    
        Totalhouseincome = np.zeros(len(datatable2))
    
        WeightedIncome = np.zeros(len(datatable2))
        WeightedIncome = (mean1*datatable2[:,0]+mean2*datatable2[:,1]+mean3*datatable2[:,2]+mean4*datatable2[:,3]
                    +mean5*datatable2[:,4]+mean6*datatable2[:,5]+mean7*datatable2[:,6]+mean8*datatable2[:,7]
                    +mean9*datatable2[:,8]+mean10*datatable2[:,9]+mean11*datatable2[:,10]+mean12*datatable2[:,11]
                    +mean13*datatable2[:,12]+mean14*datatable2[:,13]+mean15*datatable2[:,14]+mean16*datatable2[:,15])
 
        DistributionTable2 = np.zeros((len(datatable2),16))
        for i in range(len(datatable2)):    
            Totalhouseincome[i] = np.sum(datatable2[i])
            DistributionTable2[i,:] = datatable2[i,:]/Totalhouseincome[i]
    
        Average_householdincome = WeightedIncome/Totalhouseincome
        Average_householdincome = np.nan_to_num(Average_householdincome)
        DistributionTable2 = np.nan_to_num(DistributionTable2) 
    
        Blockgroup2 = np.array([np.append(DistributionTable2[i], Average_householdincome[i]) for i in range(len(datatable))])    
        bblockgroup2=Blockgroup2[~np.all(Blockgroup2 == 0, axis=1)]
    
        x3=np.array(df2)
        TAZtable2 = x3.astype(np.float)
        TAZ2 = np.zeros((len(TAZtable2),16))
        idx2 = np.zeros(len(TAZtable2))
        TAZ_sum = np.zeros(len(TAZtable2))
        Total_TAZ = np.zeros((len(TAZtable2),17))
        Final_form = np.zeros((len(TAZ2),18))
    
        for i in range(len(TAZtable2)):   
            idx2[i] = np.argmin(abs(TAZtable2[i,3]-bblockgroup2[:,16]))
            TAZ2[i,:] = TAZtable2[i,1]*bblockgroup2[int(idx2[i]),:-1]
            TAZ_sum[i] = sum(TAZ2[i,:])
            Total_TAZ[i,:] = np.append(TAZ_sum[i], TAZ2[i,:])
    
        for i in range(len(TAZtable2)):
            Final_form[i,:] = np.append((TAZtable2[i,0]),(Total_TAZ[i,:]))
    
    
    
    
        # To set up work space
        address_data = np.zeros((len(datatable),householdnum))
        i=0
        for i in range(len(address_data)):
            address_data[i,0] = datatable[i,0]
            address_data[i,1] = datatable[i,1]+datatable[i,2]
            address_data[i,2] = datatable[i,3]+datatable[i,4]
            address_data[i,3] = datatable[i,5]+datatable[i,6]
            address_data[i,4] = datatable[i,7]+datatable[i,8]
            address_data[i,5] = datatable[i,9]+datatable[i,10]
            address_data[i,6] = datatable[i,11]+datatable[i,12]
    
        # To derive block group
        Totalhousenum = np.zeros(len(address_data))
        Totalpopnum = np.zeros(len(address_data))
        DistributionTable = np.zeros((len(address_data),householdnum))
        # To include TAZ ID number, we increase row number
        Blockgroup = np.zeros((len(address_data),householdnum+1))
        i=0
        for i in range(len(address_data)):    
            Totalhousenum[i] = np.sum(address_data[i])
            DistributionTable[i,:] = address_data[i,:]/Totalhousenum[i]
        # Total population caluclation by each household size
        Totalpopnum = 1*address_data[:,0]+2*address_data[:,1]+3*address_data[:,2]+4*address_data[:,3]+5*address_data[:,4]+6*address_data[:,5]+Avg_7more*address_data[:,6]
        Average_householdsize = Totalpopnum/Totalhousenum
    
        #Converting Nan values into zero values
        Average_householdsize = np.nan_to_num(Average_householdsize)
        DistributionTable = np.nan_to_num(DistributionTable) 
    
        #Loop for Blockgroup appending distribution information
        Blockgroup = np.array([np.append(DistributionTable[i], Average_householdsize[i]) for i in range(len(address_data))])    
        bblockgroup=Blockgroup[~np.all(Blockgroup == 0, axis=1)]
    
        # Load TAZ table
        x4=np.array(df2)
        TAZtable = x4.astype(np.float)
        TAZ = np.zeros((len(TAZtable),householdnum))
        i=0
        idx = np.zeros(len(TAZtable))
        for i in range(len(TAZtable)):   
            idx[i] = np.argmin(abs(TAZtable[i,2]-bblockgroup[:,7]))
            TAZ[i,:] = TAZtable[i,1]*bblockgroup[int(idx[i]),:-1]
    
    
        Final=np.hstack((Final_form,TAZ))
    
        return Final
    
    def Group_Marginal(self):       
        data = data3
        data = data.sort_values(by='bpm2012TAZ')
        group = pd.DataFrame({'geo':data['bpm2012TAZ'], 'gqtotal':data['GQPOP12']
                       ,'gqtype1':data['GQPOPINS12'],'gqtype2':data['GQPOPSTR12']+data['GQPOPOTH12']})

        # To change the order of column title
        group = group[['geo','gqtotal','gqtype1','gqtype2']]

        # Length of each state TAZ
        newyork = len(data[data['STATE']==36])
        newjersey = len(data[data['STATE']==34])
        connecticut = len(data[data['STATE']==9])
    
        # Assign each TAZ group
        Newyork = group.iloc[0:newyork]
        Newjersey = group.iloc[newyork:newjersey+newyork]
        Connecticut = group.iloc[newjersey+newyork:connecticut+newjersey+newyork]
    
        return Newyork, Newjersey, Connecticut
    
    def Person_Marginal(self):      
        data = data3
        data = data.sort_values(by='bpm2012TAZ')
        person = pd.DataFrame({'geo':data['bpm2012TAZ'], 'pworker1':data['ELF12']
                       ,'pworker2':data['TOTPOP12']-data['ELF12'], 'ptotal':data['TOTPOP12']})

        # To change the order of column title
        person = person[['geo','pworker1','pworker2','ptotal']]

        # Each state TAZ last numbers
        newyork = len(data[data['STATE']==36])
        newjersey = len(data[data['STATE']==34])
        connecticut = len(data[data['STATE']==9])
    
        # Assign each TAZ group
        Newyork = person.iloc[0:newyork]
        Newjersey = person.iloc[newyork:newjersey+newyork]
        Connecticut = person.iloc[newjersey+newyork:connecticut+newjersey+newyork]
    
        return Newyork, Newjersey, Connecticut

data1 = pd.read_csv('Input CSVs/ACS_14_5YR_B11016_with_ann.csv',low_memory=False)
data2 = pd.read_csv('Input CSVs/ACS_14_5YR_B19001_with_ann.csv',low_memory=False)
data3 = pd.read_excel('Input CSVs/NYBPM2012_SED_2050Series_V3_May2017_.xlsx') 
data4 = pd.read_csv('Input CSVs/ss12hct.csv')
data5 = pd.read_csv('Input CSVs/ss12hny.csv')
data6 = pd.read_csv('Input CSVs/ss12hnj.csv')

# New York marginal files
Newyork_path = r'./Newyork' 
if not os.path.exists(Newyork_path):
     os.makedirs(Newyork_path)

# To derive table frame for PopGen
with open('Newyork/household_marginals.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhtotals","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc",
                     "hsize","hsize","hsize","hsize","hsize","hsize","hsize"])
    writer.writerow(["variable_categories","","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "1","2","3","4","5","6","7"])
    writer.writerow(["geo"])

Newyork_dic = {'New York':7.66,'Queens':8.09, 'Bronx':7.683, 'Kings':8.08, 'Richmond':7.76, 'Nassau':7.87,
              'Suffolk':8.01,'Westchester':7.384,'Rockland':8.6, 'Putnam':7.5, 'Orange':8.11, 'Dutchess':8.28}
for key,value in Newyork_dic.items():
        HHdistributionTable = pd.DataFrame(Marginals(data1,data2,data3,data4,data5,data6).Hsize_Distribution(key,value))  
        HHdistributionTable.to_csv('Newyork/household_marginals.csv',mode='a',index=False, header=False)

with open('Newyork/groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","gqtotals","gqtype","gqtype"])
    writer.writerow(["variable_categories","1","1","2"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Group_Marginal()[0].to_csv('Newyork/groupquarters_marginals.csv', mode='a', index=False, header=False)
 
with open('Newyork/person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","pworker","pworker","ptotals"])
    writer.writerow(["variable_categories","1","2","1"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Person_Marginal()[0].to_csv('Newyork/person_marginals.csv', mode='a', index=False, header=False)

# New Jersey marginal files
Newjersey_path = r'./Newjersey' 
if not os.path.exists(Newjersey_path):
     os.makedirs(Newjersey_path)

# To derive table frame for PopGen
with open('Newjersey/household_marginals.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhtotals","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc",
                     "hsize","hsize","hsize","hsize","hsize","hsize","hsize"])
    writer.writerow(["variable_categories","","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "1","2","3","4","5","6","7"])
    writer.writerow(["geo"])

Newjersey_dic = {'Bergen':7.82, 'Passaic':7.78, 'Hudson':7.92, 'Essex':7.63, 'Union':7.64, 'Morris':8.54,
                 'Somerset':8.25, 'Middlesex':7.69,'Monmouth':7.45, 'Ocean':7.9, 'Hunterdon':7, 'Warren':8,
                 'Sussex':7.91, 'Mercer':7.6}

for key,value in Newjersey_dic.items():
        HHdistributionTable = pd.DataFrame(Marginals(data1,data2,data3,data4,data5,data6).Hsize_Distribution(key,value))  
        HHdistributionTable.to_csv('Newjersey/household_marginals.csv',mode='a',index=False, header=False)

with open('Newjersey/groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","gqtotals","gqtype","gqtype"])
    writer.writerow(["variable_categories","1","1","2"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Group_Marginal()[1].to_csv('Newjersey/groupquarters_marginals.csv', mode='a', index=False, header=False)
 
with open('Newjersey/person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","pworker","pworker","ptotals"])
    writer.writerow(["variable_categories","1","2","1"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Person_Marginal()[1].to_csv('Newjersey/person_marginals.csv', mode='a', index=False, header=False)

# Connecticut marginal files
Connecticut_path = r'./Connecticut' 
if not os.path.exists(Connecticut_path):
     os.makedirs(Connecticut_path)
    
# To derive table frame for PopGen
with open('Connecticut/household_marginals.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhtotals","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc","hinc",
                     "hsize","hsize","hsize","hsize","hsize","hsize","hsize"])
    writer.writerow(["variable_categories","","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "1","2","3","4","5","6","7"])
    writer.writerow(["geo"])

Connecticut_dic = {'Fairfield':7.5, 'New Haven':7.74}

for key,value in Connecticut_dic.items():
        HHdistributionTable = pd.DataFrame(Marginals(data1,data2,data3,data4,data5,data6).Hsize_Distribution(key,value))  
        HHdistributionTable.to_csv('Connecticut/household_marginals.csv',mode='a',index=False, header=False)

with open('Connecticut/groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","gqtotals","gqtype","gqtype"])
    writer.writerow(["variable_categories","1","1","2"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Group_Marginal()[2].to_csv('Connecticut/groupquarters_marginals.csv', mode='a', index=False, header=False)

with open('Connecticut/person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["varialbe_names","pworker","pworker","ptotals"])
    writer.writerow(["variable_categories","1","2","1"])
    writer.writerow(["geo"])
Marginals(data1,data2,data3,data4,data5,data6).Person_Marginal()[2].to_csv('Connecticut/person_marginals.csv', mode='a', index=False, header=False)


# In[3]:


class Input_samples():
    
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    
    def groupquarter_samples(self,data1): 
    
        # NP 2 and NP 3 groupquarters
        GQ_TYPE = self.data1[(data1.TYPE == 2) | (data1.TYPE == 3)]

        GQ_sample = pd.DataFrame({'hid':GQ_TYPE['SERIALNO'], 'sample_geo':GQ_TYPE['PUMA'],
                                  'gqtype':GQ_TYPE['TYPE']})
        GQ_sample = GQ_sample[['hid','sample_geo','gqtype']]
        GQ_sample.loc[GQ_sample.gqtype == 2, 'gqtype'] = 1
        GQ_sample.loc[GQ_sample.gqtype == 3, 'gqtype'] = 2
        GQ_sample['gqrtotals']=1
        GQ_sample['gqtototals']=1 
    
        return GQ_sample
    
    def household_samples(self,data1):
    
        # NP 2 and NP 3 groupquarters
        HH_TYPE = self.data1[(data1.TYPE == 1) & (data1.NP > 0)]
        HH_sample = pd.DataFrame({'hid':HH_TYPE['SERIALNO'],'sample_geo':HH_TYPE['PUMA'],
                                         'hinc':HH_TYPE['HINCP'], 'hsize':HH_TYPE['NP']})
        HH_sample = HH_sample[['hid','sample_geo','hinc','hsize']]

            # 7 or more Household size is equal to 7 
        HH_sample.hsize[HH_sample.hsize > 7] = 7

        # Categorizing 16 different income classes
        HH_sample.loc[HH_sample.hinc < 10000, 'hinc'] = 1
        HH_sample.loc[(HH_sample.hinc >= 10000) & (HH_sample.hinc < 15000), 'hinc'] = 2
        HH_sample.loc[(HH_sample.hinc >= 15000) & (HH_sample.hinc < 20000), 'hinc'] = 3
        HH_sample.loc[(HH_sample.hinc >= 20000) & (HH_sample.hinc < 25000), 'hinc'] = 4
        HH_sample.loc[(HH_sample.hinc >= 25000) & (HH_sample.hinc < 30000), 'hinc'] = 5
        HH_sample.loc[(HH_sample.hinc >= 30000) & (HH_sample.hinc < 35000), 'hinc'] = 6
        HH_sample.loc[(HH_sample.hinc >= 35000) & (HH_sample.hinc < 40000), 'hinc'] = 7
        HH_sample.loc[(HH_sample.hinc >= 40000) & (HH_sample.hinc < 45000), 'hinc'] = 8
        HH_sample.loc[(HH_sample.hinc >= 45000) & (HH_sample.hinc < 50000), 'hinc'] = 9
        HH_sample.loc[(HH_sample.hinc >= 50000) & (HH_sample.hinc < 60000), 'hinc'] = 10
        HH_sample.loc[(HH_sample.hinc >= 60000) & (HH_sample.hinc < 75000), 'hinc'] = 11
        HH_sample.loc[(HH_sample.hinc >= 75000) & (HH_sample.hinc < 100000), 'hinc'] = 12
        HH_sample.loc[(HH_sample.hinc >= 100000) & (HH_sample.hinc < 125000), 'hinc'] = 13
        HH_sample.loc[(HH_sample.hinc >= 125000) & (HH_sample.hinc < 150000), 'hinc'] = 14
        HH_sample.loc[(HH_sample.hinc >= 150000) & (HH_sample.hinc < 200000), 'hinc'] = 15
        HH_sample.loc[(HH_sample.hinc >= 200000), 'hinc'] = 16
        HH_sample = HH_sample.astype(int)
        HH_sample['hhrtotals']=1
        HH_sample['hhtototals']=1
    
        return HH_sample

    def person_samples(self,data2):

        person_sample = pd.DataFrame({'hid':data2['SERIALNO'], 'pid':data2['SPORDER'], 'sample_geo':data2['PUMA'],
                                         'pworker':data2['ESR'],'rpworker':data2['ESR'],
                                         'rpage':data2['AGEP'],'rpsex':data2['SEX']})
        person_sample['prtotals']=1
        person_sample['ptotals']=1

        person_sample = person_sample[['hid','pid','sample_geo','ptotals','prtotals','pworker',
                                                  'rpworker','rpage','rpsex']]
        person_sample['pworker'].fillna(0, inplace=True)
        person_sample.loc[(person_sample.pworker ==1)|(person_sample.pworker ==2)|
                               (person_sample.pworker ==4)|(person_sample.pworker ==5),
                               'pworker'] = 1
        person_sample.loc[(person_sample.pworker ==0)|(person_sample.pworker ==3)|
                               (person_sample.pworker ==6),
                              'pworker'] = 2
        person_sample['rpworker'] = person_sample['pworker']

        person_sample.loc[(person_sample.rpage < 5), 'rpage'] = 1
        person_sample.loc[(person_sample.rpage >= 5) & (person_sample.rpage < 10), 'rpage'] = 2
        person_sample.loc[(person_sample.rpage >= 10) & (person_sample.rpage < 15), 'rpage'] = 3
        person_sample.loc[(person_sample.rpage >= 15) & (person_sample.rpage < 20), 'rpage'] = 4
        person_sample.loc[(person_sample.rpage >= 20) & (person_sample.rpage < 25), 'rpage'] = 5
        person_sample.loc[(person_sample.rpage >= 25) & (person_sample.rpage < 30), 'rpage'] = 6
        person_sample.loc[(person_sample.rpage >= 30) & (person_sample.rpage < 35), 'rpage'] = 7
        person_sample.loc[(person_sample.rpage >= 35) & (person_sample.rpage < 40), 'rpage'] = 8
        person_sample.loc[(person_sample.rpage >= 40) & (person_sample.rpage < 45), 'rpage'] = 9
        person_sample.loc[(person_sample.rpage >= 45) & (person_sample.rpage < 50), 'rpage'] = 10
        person_sample.loc[(person_sample.rpage >= 50) & (person_sample.rpage < 55), 'rpage'] = 11
        person_sample.loc[(person_sample.rpage >= 55) & (person_sample.rpage < 60), 'rpage'] = 12
        person_sample.loc[(person_sample.rpage >= 60) & (person_sample.rpage < 65), 'rpage'] = 13
        person_sample.loc[(person_sample.rpage >= 65) & (person_sample.rpage < 70), 'rpage'] = 14
        person_sample.loc[(person_sample.rpage >= 70) & (person_sample.rpage < 75), 'rpage'] = 15
        person_sample.loc[(person_sample.rpage >= 75) & (person_sample.rpage < 80), 'rpage'] = 16
        person_sample.loc[(person_sample.rpage >= 80) & (person_sample.rpage < 85), 'rpage'] = 17
        person_sample.loc[(person_sample.rpage >= 85), 'rpage'] = 18
        person_sample = person_sample.astype(int)

        return person_sample

# Connecticut sample files
ct1 = pd.read_csv('Input CSVs/ss12hct.csv')
ct2 = pd.read_csv('Input CSVs/ss12pct.csv')
Connecticut_path = r'./Connecticut' 
if not os.path.exists(Connecticut_path):
     os.makedirs(Connecticut_path)
    
with open('Connecticut/groupquarter_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","gqtype","gqrtotals","gqtotals"])
Input_samples(ct1,ct2).groupquarter_samples(ct1).to_csv('Connecticut/groupquarter_sample.csv', mode='a', index=False, header=False)
    
with open('Connecticut/household_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","hinc","hsize","hhrtotals","hhtotals"])
Input_samples(ct1,ct2).household_samples(ct1).to_csv('Connecticut/household_sample.csv', mode='a', index=False, header=False)

with open('Connecticut/person_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","pid","sample_geo","ptotals","prtotals","pworker","rpworker","rpage","rpsex"])
Input_samples(ct1, ct2).person_samples(ct2).to_csv('Connecticut/person_sample.csv', mode='a', index=False, header=False)

# New York sample files
ny1 = pd.read_csv('Input CSVs/ss12hny.csv')
ny2 = pd.read_csv('Input CSVs/ss12pny.csv')
NewYork_path = r'./NewYork' 
if not os.path.exists(NewYork_path):
     os.makedirs(NewYork_path)
    
with open('NewYork/groupquarter_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","gqtype","gqrtotals","gqtotals"])
Input_samples(ny1,ny2).groupquarter_samples(ny1).to_csv('NewYork/groupquarter_sample.csv', mode='a', index=False, header=False)
    
with open('NewYork/household_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","hinc","hsize","hhrtotals","hhtotals"])
Input_samples(ny1,ny2).household_samples(ny1).to_csv('NewYork/household_sample.csv', mode='a', index=False, header=False)

with open('NewYork/person_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","pid","sample_geo","ptotals","prtotals","pworker","rpworker","rpage","rpsex"])
Input_samples(ny1, ny2).person_samples(ny2).to_csv('NewYork/person_sample.csv', mode='a', index=False, header=False)

# New Jersey sample files
nj1 = pd.read_csv('Input CSVs/ss12hnj.csv')
nj2 = pd.read_csv('Input CSVs/ss12pnj.csv')

NewJersey_path = r'./NewJersey' 
if not os.path.exists(NewJersey_path):
     os.makedirs(NewJersey_path)
    
with open('NewJersey/groupquarter_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","gqtype","gqrtotals","gqtotals"])
Input_samples(nj1,nj2).groupquarter_samples(nj1).to_csv('NewJersey/groupquarter_sample.csv', mode='a', index=False, header=False)
    
with open('NewJersey/household_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","sample_geo","hinc","hsize","hhrtotals","hhtotals"])
Input_samples(nj1,nj2).household_samples(nj1).to_csv('NewJersey/household_sample.csv', mode='a', index=False, header=False)

with open('NewJersey/person_sample.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["hid","pid","sample_geo","ptotals","prtotals","pworker","rpworker","rpage","rpsex"])
Input_samples(nj1, nj2).person_samples(nj2).to_csv('NewJersey/person_sample.csv', mode='a', index=False, header=False)


# In[5]:


class region_marginal():
    
    def __init__(self, data1, data2, data3, data4, data5, countyname, regionID):
        
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5
        self.countyname = countyname
        self.regionID = regionID
        
    def region_GQ(self):
        
        Countyname = self.countyname
        RegionID = self.regionID
        # Delete 'NaN' in the cells
        GQ = data1
        GQ = GQ.dropna(how='all')
        GQ.reset_index(drop=True)
        # Set the index 
        GQ = GQ.set_index('POPULATION IN GROUP QUARTERS')
        # To convert header by year
        new_header1 = GQ.iloc[0] 
        GQ = GQ[1:] 
        GQ.columns = new_header1
        # Provide new index 
        GQ = GQ.reset_index()
        # Search for county
        Location1 = GQ[GQ['POPULATION IN GROUP QUARTERS'].str.contains(Countyname)]
        Region_GQ = Location1[2012].values     
        ID = np.array([RegionID])
        gq_table = np.append([ID], [Region_GQ], axis=1)
        
        return gq_table
    
    def region_HH(self):
        
        Countyname = self.countyname
        RegionID = self.regionID
        # Delete 'NaN' in the cells
        HH = data2
        HH = HH.dropna(how='all')
        HH.reset_index(drop=True)
        # Set the index 
        HH = HH.set_index('HOUSEHOLDS')
        # To convert header by year
        new_header2 = HH.iloc[0] 
        HH = HH[1:] 
        HH.columns = new_header2
        # Provide new index 
        HH = HH.reset_index()
        # Search for county
        Location2 = HH[HH['HOUSEHOLDS'].str.contains(Countyname)]
        
        # Base year 
        Region_HH = Location2[2012].values     
        ID = np.array([RegionID])
        hh_table = np.append([ID], [Region_HH], axis=1)
        
        return hh_table
    
    def region_person(self):

        Countyname = self.countyname
        RegionID = self.regionID    
        # Delete 'NaN' in the cells
        Tot = data3
        Tot = Tot.dropna(how='all')
        Tot.reset_index(drop=True)
        # Set the index 
        Tot = Tot.set_index('TOTAL POPULATION')
        # To convert header by year
        new_header3 = Tot.iloc[0] 
        Tot = Tot[1:] 
        Tot.columns = new_header3
        # Provide new index 
        Tot = Tot.reset_index()
        # Search for county
        Location3 = Tot[Tot['TOTAL POPULATION'].str.contains(Countyname)]
        Tot_Pop = Location3[2012].values 
               
        # Delete 'NaN' in the cells
        Emp = data4
        Emp = Emp.dropna(how='all')
        Emp.reset_index(drop=True)
        # Set the index 
        Emp = Emp.set_index('EMPLOYED LABOR FORCE')
        # To convert header by year
        new_header4 = Emp.iloc[0] 
        Emp = Emp[1:] 
        Emp.columns = new_header4
        # Provide new index 
        Emp = Emp.reset_index()
        # Search for county
        Location4 = Emp[Emp['EMPLOYED LABOR FORCE'].str.contains(Countyname)]
        Employed = Location4[2012].values
    
        # Delete 'NaN' in the cells
        Distribution = data5.dropna(how='all')
        Distribution.reset_index(drop=True)
        # Set the index 
        Distribution = Distribution.set_index('TOTAL POPULATION')
        # To convert header by year
        new_header5 = Distribution.iloc[0] 
        Distribution = Distribution[1:] 
        Distribution.columns = new_header5
        # Provide new index 
        Distribution = Distribution.reset_index()
        # Search for county
        Location5 = Distribution[Distribution['TOTAL POPULATION'].str.contains(Countyname)]
        age_distribution2010=Location5[2010].iloc[:,0].values
        age_distribution2015=Location5[2015].iloc[:,0].values

        man_2010 = Location5[2010].iloc[-1,1]
        man_2015 = Location5[2015].iloc[-1,1]
        mean_man = np.mean([man_2010,man_2015])

        women_2010 = Location5[2010].iloc[-1,2]
        women_2015 = Location5[2015].iloc[-1,2]
        mean_women = np.mean([women_2010,women_2015])

        average_between = np.mean([age_distribution2010, age_distribution2015],axis=0)

        # 2012 is base year
        Unemployed = Tot_Pop - Employed
        Age = average_between[:-1]*Tot_Pop/average_between[-1]
        Man = mean_man*Tot_Pop/average_between[-1]
        Women = mean_women*Tot_Pop/average_between[-1]
        ID = np.array([RegionID])
        Combined_values = np.concatenate((Man,Women, Age, Employed, Unemployed, Tot_Pop),axis=0)

        ID = np.array([RegionID])
        Person_table = np.append([ID], [Combined_values], axis=1)
        
        return Person_table

data1 = pd.read_excel('Input CSVs/2050 SED County Level Forecasts - Detailed Tables.xlsx', sheet_name='Population in Group Quarters')
data2 = pd.read_excel('Input CSVs/2050 SED County Level Forecasts - Detailed Tables.xlsx', sheet_name='Households')
data3 = pd.read_excel('Input CSVs/2050 SED County Level Forecasts - Detailed Tables.xlsx', sheet_name='Total Population')
data4 = pd.read_excel('Input CSVs/2050 SED County Level Forecasts - Detailed Tables.xlsx', sheet_name='Employed Labor Force')
data5 = pd.read_excel('Input CSVs/2050 SED County Level Forecasts - Detailed Tables.xlsx', sheet_name='Total Population Detail')

Newyork_path = r'./Newyork' 
Newyork_dic2 = {'New York':1,'Queens':2, 'Bronx':3, 'Kings':4, 'Richmond':5, 'Nassau':6,
              'Suffolk':7,'Westchester':8,'Rockland':9, 'Putnam':10, 'Orange':11, 'Dutchess':12}

if not os.path.exists(Newyork_path):
     os.makedirs(Newyork_path)    

with open('Newyork/region_groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","gqrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])
for key,value in Newyork_dic2.items():
    rGQ_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_GQ())  
    rGQ_table.to_csv('Newyork/region_groupquarters_marginals.csv',mode='a',index=False, header=False)

with open('Newyork/region_household_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])
for key,value in Newyork_dic2.items():
        rHH_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_HH())  
        rHH_table.to_csv('Newyork/region_household_marginals.csv',mode='a',index=False, header=False)

with open('Newyork/region_person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","rpsex","rpsex","rpage","rpage","rpage","rpage","rpage","rpage","rpage",
                    "rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage", "rpworker",
                    "rpworker","prtotals"])
    writer.writerow(["variable_categories","1","2","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "17","18","1","2","1"])
    writer.writerow(["region"])
for key,value in Newyork_dic2.items():
        Person_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_person())  
        Person_table.to_csv('Newyork/region_person_marginals.csv',mode='a',index=False, header=False)    

# NewJersey
Newjersey_path = r'./Newjersey' 
if not os.path.exists(Newjersey_path):
     os.makedirs(Newjersey_path)
    
with open('Newjersey/region_groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","gqrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])

Newjersey_dic2 = {'Bergen':1,'Passaic':2, 'Hudson':3, 'Essex':4, 'Union':5, 'Morris':6,
              'Somerset':7,'Middlesex':8,'Monmouth':9, 'Ocean':10, 'Hunterdon':11, 'Warren':12, 'Sussex':13, 'Mercer':14}

for key,value in Newjersey_dic2.items():
        rGQ_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_GQ())  
        rGQ_table.to_csv('Newjersey/region_groupquarters_marginals.csv',mode='a',index=False, header=False)
        
with open('Newjersey/region_household_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])
    
for key,value in Newjersey_dic2.items():
        rHH_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_HH())  
        rHH_table.to_csv('Newjersey/region_household_marginals.csv',mode='a',index=False, header=False)
        
with open('Newjersey/region_person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","rpsex","rpsex","rpage","rpage","rpage","rpage","rpage","rpage","rpage",
                    "rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage", "rpworker",
                    "rpworker","prtotals"])
    writer.writerow(["variable_categories","1","2","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "17","18","1","2","1"])
    writer.writerow(["region"])
    
for key,value in Newjersey_dic2.items():
        Person_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_person())  
        Person_table.to_csv('Newjersey/region_person_marginals.csv',mode='a',index=False, header=False)

# Connecticut
Connecticut_path = r'./Connecticut' 
if not os.path.exists(Connecticut_path):
     os.makedirs(Connecticut_path)

Connecticut_dic2 = {'Fairfield':1, 'New Haven':2}

with open('Connecticut/region_groupquarters_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","gqrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])

for key,value in Connecticut_dic2.items():
        rGQ_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_GQ())  
        rGQ_table.to_csv('Connecticut/region_groupquarters_marginals.csv',mode='a',index=False, header=False)

with open('Connecticut/region_household_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["variable_names","hhrtotals"])
    writer.writerow(["variable_categories","1"])
    writer.writerow(["region"])

for key,value in Connecticut_dic2.items():
        rHH_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_HH())  
        rHH_table.to_csv('Connecticut/region_household_marginals.csv',mode='a',index=False, header=False)    
    
with open('Connecticut/region_person_marginals.csv','w', newline='') as outcsv:
    writer = csv.writer(outcsv) 
    writer.writerow(["variable_names","rpsex","rpsex","rpage","rpage","rpage","rpage","rpage","rpage","rpage",
                    "rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage","rpage", "rpworker",
                    "rpworker","prtotals"])
    writer.writerow(["variable_categories","1","2","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                    "17","18","1","2","1"])
    writer.writerow(["region"])

for key,value in Connecticut_dic2.items():
        Person_table = pd.DataFrame(region_marginal(data1,data2,data3,data4,data5, key, value).region_person())  
        Person_table.to_csv('Connecticut/region_person_marginals.csv',mode='a',index=False, header=False)  

