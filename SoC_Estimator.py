"""
Opening LG files and cleaning data

References: 
- https://python-forum.io/thread-26450-page-2.html
- Batch sizes: https://stackoverflow.com/questions/74631843/training-and-validation-loss-history-for-mlpregressor
"""

import numpy as np
import pandas as pd

class SoC_Estimator():
    def __init__(self, path):
        self.path = path
        
    def read_data(self):
        # Read file and drop empty cols/rows
        df = pd.read_csv(self.path, skiprows=[i for i in range(28)])
        df = df.drop(0)
        df = df.drop(df.columns[-1],axis=1)

        # Convert relevant cols to floats
        df[['Voltage', 'Current', 'Temperature', 'Capacity']] = df[['Voltage', 'Current', 'Temperature', 'Capacity']].astype(float)

        # Find largest # for SoC
        nom_cap = df[df.columns[-3]].max()

        # Create new manual SoC calculations (solution)
        df['SoC_calc'] = df[df.columns[-3]] / nom_cap

        # Loop for each cycle
        start = True
        end = False
        cycle_starts = []
        cycle_ends = []
        cycle_num = 1
        cycle_nums = []

        for row_num in range(len(df)):
            # print("CYCLE_NUM: ", cycle_num)
            # print("ROW_NUM: ", row_num)
            # print("df.iloc[row_num, -6]: ", df.iloc[row_num, -6])
            if (df.iloc[row_num, -6] == float(0) and start):
                cycle_starts.append(row_num)
                start = False
                cycle_nums.append(cycle_num)
            elif (df.iloc[row_num, -6] > float(0) and not start):
                end = True
                cycle_nums.append(cycle_num)
            elif (df.iloc[row_num, -6] == float(0) and end):
                cycle_ends.append(row_num)
                start = True
                end = False
                cycle_nums.append(cycle_num)
                cycle_num += 1
            else:
                cycle_nums.append(cycle_num)

        # cycles_df = pd.DataFrame({'Cycle':cycle_nums})
        # print("CYCLES_DF: ", cycles_df)
        # df = df.join(cycles_df)
        print("BEFORE ADDING CYCLE: --------------------: ", df)
        print("CYCLE_NUM length: ", len(cycle_nums))
        df['Charging_Cycle'] = cycle_nums

        self.data = df
        self.cycle_starts = cycle_starts
        self.cycle_ends = cycle_ends

        return df, cycle_starts, cycle_ends
    
    def read_discharge_data(self):
        # Read file and drop empty cols/rows
        df = pd.read_csv(self.path)
        # df = df.drop(0)
        # df = df.drop(df.columns[-1],axis=1)
        print(df)

        # Convert relevant cols to floats
        df[['Voltage', 'Current', 'Temperature', 'Capacity']] = df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Capacity']].astype(float)
        print("DF['VOLTAGE]", df['Voltage'])
        print("DF['CURRENT]", df['Current'])
        print("DF['TEMP]", df['Temperature'])
        print("DF['CAPACITY]", df['Capacity'])

        ## FIND CYCLES
        prev = df['Time'][0]
        cycle_starts = [0]
        cycle_ends = []
        cycle_num = 1
        cycle_nums = [1]

        for i in range(1, len(df['Time'])):
            if prev > df['Time'][i]:
                cycle_starts.append(i)
                cycle_ends.append(i - 1)
                cycle_num += 1
                cycle_nums.append(cycle_num)
            else:
                cycle_nums.append(cycle_num)
            prev = df['Time'][i]
        cycle_starts.pop()
        print("CYCLE NUM: ", cycle_num)

        print("df[capacity]: ", df['Capacity'])

        # Find largest # for SoC
        nom_cap = df['Capacity'].max()

        print("NOM_CAP: ", nom_cap)

        # Create new manual SoC calculations (solution)
        df['SoC_calc'] = df['Capacity'] / nom_cap

        print(df)

        # cycles_df = pd.DataFrame({'Cycle':cycle_nums})
        # print("CYCLES_DF: ", cycles_df)
        # df = df.join(cycles_df)
        print("BEFORE ADDING CYCLE: --------------------: ", df)
        print("CYCLE_NUM length: ", len(cycle_nums))
        print("CYCLE_STARTS: ", cycle_starts)
        print("CYCLE_ENDS: ", cycle_ends)

        print("DF LENGTH: ", len(df['Time']))
        print("CYCLE_NUMS LENGTH: ", len(cycle_nums))

        df['Charging_Cycle'] = cycle_nums

        self.data = df
        self.cycle_starts = cycle_starts
        self.cycle_ends = cycle_ends

        return df, cycle_starts, cycle_ends
    
    # def cycle(self):
    #     start = True
    #     cycle_starts = []
    #     cycle_ends = []

    #     for row_num in range(len(self.df)):
    #         print("ROW_NUM: ", row_num)
    #         print("df.iloc[row_num, -5]: ", df.iloc[row_num, -5])
    #         if (start):
    #             cycle_starts.append(row_num)
    #             start = False
    #         elif (df.iloc[row_num, -5] > float(0) and not start):
    #             start = True
    #         elif (df.iloc[row_num, -5] == float(0) and start):
    #             cycle_ends.append(row_num)
    #             start = False

    #     return cycle_starts, cycle_ends

    def mlp_regressor(self):
        print("hi")

if __name__ == '__main__':
    # Provide path and filename from outside
    path = 'cp3473x7xv-3/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/0degC/585_Dis_0p5C.csv'
    df = SoC_Estimator(path)

    # Params for csv_read
    df.read_data()
    df = df.data

# battery_585 = SoC_Estimator("cp3473x7xv-3/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/0degC/585_Dis_0p5C.csv")
# test = battery_585.read_data()
# print("PRINTING TEST -----------------")
# print(test) ### EDIT -- add to instance ?