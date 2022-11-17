import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = []
with open('Market_Basket_Optimisation.csv') as file:     #cant read file using Pandas
    reader = csv.reader(file, delimiter=',')             #reading csv file (with open)
    for row in reader:
        dataset += [row]
        
        
print(dataset)


#Unstuctured to structured
te = TransactionEncoder()
x = te.fit_transform(dataset)

df = pd.DataFrame(x, columns= te.columns_)          #Assigning Food Objects to column names
print(te.columns_)

#Finding Frequent Itemsets
freq_itemset = apriori(df, min_support=0.01, use_colnames=True)
print(freq_itemset)

#Find the rules
rules = association_rules(freq_itemset, metric='confidence', min_threshold=0.25)

rules = rules[['antecedents','consequents','support','confidence']]      
print(rules.head())

suggestion= rules[rules['antecedents'] == {'burgers'}]['consequents'] #predicitng or suggesting an item
print(suggestion)