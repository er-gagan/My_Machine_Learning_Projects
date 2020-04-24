#=======================Some Important Module Imported===========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
from termcolor import cprint
#=========================Functions Define=========================================
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file in the directory'
    else:
        return csv_files
def display_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print("Press ",end="")
        cprint('%d'%i, 'red', attrs=['bold'], file=sys.stderr,end="")
        print(" to select ",end="")
        cprint('%s'%file_name, 'green', attrs=['bold'], file=sys.stderr)
        i+=1
    return csv_files[int(input("\nSelect file to create ML model "))]
def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best Fit')
    plt.scatter(X_test,Y_test,color='green',label='test data')
    plt.scatter(X_test,Y_pred,color='black',label='Pred test data')
    plt.title("Salary vs Experience")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    
    plt.legend()
    plt.show()
#=============================main Function Definition Started=================================
def main():
    os.system("cls")
    input("Welcome to Salary Prediction System\nPress ENTER key to proceed..\n")
    try:
        csv_files=checkcsv()
        if csv_files=='No csv file in the directory':
            raise FileNotFoundError('No csv file in the directory')
        csv_file=display_and_select_csv(csv_files)
        cprint('%s'%csv_file, 'green', attrs=['bold'], file=sys.stderr,end="")
        print(' is selected')
        print('\nReading csv file..\nCreating Dataset..')
        dataset=pd.read_csv(csv_file)
        print('Dataset created..\n')
        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        s=float(input("Enter test data size (between 0 and 1) ") )
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
        
        print("\nModel creation in progression")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        input("Model is created\n\nPress any key to predict test data in trained model..")
        print("---------------------------------------------------\n\n")
        Y_pred=regressionObject.predict(X_test)
        i=0
        print("X_test          Y_test          Y_pred\n")
        while i<len(X_test):
            print(X_test[i],'        ',Y_test[i],'        ',Y_pred[i])
            i+=1
        input("\nPress ENTER key to see above result in graphical format.. ")
        graph(X_train,Y_train, regressionObject, X_test, Y_test, Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("\nOur model is %2.2f%% accurate" %(r2*100))

        print("\n\nNow you can predict salary of an employee using our model")
        print("\nEnter experience in years of the candidates, separated by comma")

        exp=[float(e) for e in input().split(',')]
        ex=[]
        for x in exp:
            ex.append([x])
        experience =np.array(ex)
        salaries=regressionObject.predict(experience)

        plt.scatter(experience,salaries,color='black')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salaries')
        plt.show()

        d=pd.DataFrame({'Experience':exp,'Salaries':salaries})
        print(d)
        
    except FileNotFoundError:
        print('No csv file in the directory')
        print("Press ENTER key to exit")
        input()
        exit()

if __name__=="__main__":
    main()
    input("Press any key to exit")