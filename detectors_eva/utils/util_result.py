import pandas as pd




def write_excel(filename,sheetname,dataframe):
    with pd.ExcelWriter(filename, mode='a',if_sheet_exists='replace') as writer:# replace new
        workBook = writer.book
        # try:
        #     workBook.remove(workBook[sheetname])
        # except:
        #     print("Worksheet does not exist")
        # finally:
        dataframe.to_excel(writer, sheet_name=sheetname,index=False)
