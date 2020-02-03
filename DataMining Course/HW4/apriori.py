import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [['A', 'C'],
           ['C', 'D', 'E'],
           ['A', 'C', 'D', 'E'],
           ['D', 'E'],
           ['A', 'B', 'E'],
           ['A', 'B', 'C', 'D', 'E']]
 
def main():
    oht = OnehotTransactions()
    oht_ary = oht.fit(dataset).transform(dataset)
    df = pd.DataFrame(oht_ary, columns=oht.columns_)         
    
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    print (frequent_itemsets)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print (rules)

if __name__ == "__main__":
    main()