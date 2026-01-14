import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
@st.cache_data
def predictor():
        df = pd.read_csv('customer_purchase_data.csv')
        df_1 = pd.get_dummies(df, columns=['ProductCategory'], drop_first=True)
        X = df_1.drop('PurchaseStatus', axis=1)
        y = df_1['PurchaseStatus']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model,X.columns
model,model_columns=predictor()
def prod_id(product1):
    match product1:
        case 'electronics':
            return 0
        case 'clothing':
            return 1
        case 'Home Goods':
            return 2
        case 'Beauty':
            return 3
        case 'Sports':
            return 4
        case _:
            return None
st.title('Purchase prediction probablity checker')
age=st.number_input('Age:',min_value=18,max_value=100,value=30,step=1)
gender=st.radio('Gender:',['Male','Female'])
gender1=0 if gender=='Male' else 1
product=st.selectbox('Product Category:',['electronics','clothing','Home Goods','Beauty','Sports'])
prod=prod_id(product)
income=st.slider('Income:',min_value=0,max_value=100000,value=50000,step=10000)
purchases=st.number_input('Number of purchases:',min_value=0,max_value=100,value=1,step=1)
time=st.slider('Time Spent:',min_value=0,max_value=1000,value=1,step=10)
loyality=st.checkbox('Loyality member:',value=False)
if loyality:
    loyality=1
else:
    loyality=0
discount=st.slider('Discount Applied:',min_value=0,max_value=100,value=0,step=5)
button=st.button('Predict')
if button:
    input_data={
        'Age':[age],
        'Gender':[gender1],
        'ProductCategory':[prod],
        'AnnualIncome':[income],
        'TimeSpentOnWebsite':[time],
        'LoyaltyProgram':[loyality],
        'NumberofPurchases':[purchases],
        'DiscountsAvailed':[discount]
    }
    new_df=pd.DataFrame(input_data)
    new_df=pd.get_dummies(new_df,columns=['ProductCategory'],drop_first=True)
    new_df=new_df.reindex(columns=model_columns,fill_value=0)
    prediction=model.predict(new_df)
    probability=model.predict_proba(new_df)[0][1]
    if button:
        st.write('The probability of purchase is:')
        if prediction[0]==1:
            st.success('High')
        else:
            st.error('Low')