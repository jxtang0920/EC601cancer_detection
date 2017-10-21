#%matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
#from sklearn.feature_extraction.text import TfidfVectorizer

train_variants = pd.read_csv('training_variants.csv')
print('Number of training variants: %d' % (len(train_variants)))
print(train_variants.head())

test_variants = pd.read_csv('test_variants.csv')
print('Number of test variants: %d' % (len(test_variants)))
print(test_variants.head())

def read_textfile(filename):
    return pd.read_csv(filename, sep='\|\|', header=None, names=['ID', 'Text'], skiprows=1, engine='python')

train_text = read_textfile('training_text.csv')
print('Number of train samples: %d' % (len(train_text)))
print(train_text.head())

test_text = read_textfile('test_text.csv')
print('Number of test samples: %d' % (len(test_text)))
print(test_text.head())

train_df = pd.concat([train_text, train_variants.drop('ID', axis=1)], axis=1)
print(train_df.head())

test_df = pd.concat([test_text, test_variants.drop('ID', axis=1)], axis=1)
print(test_df.head())

#frequency of each classes
plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=train_df, palette="Reds_d")
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of genetic mutation classes", fontsize=18)
plt.show()

gene_group = train_df.groupby("Gene")['Gene'].count()
gene_group_test = test_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
minimal_occ_genes_test = gene_group_test.sort_values(ascending=True)[:10]
print("Train Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("Test Genes with maximal occurences\n", gene_group_test.sort_values(ascending=False)[:10])
print("\nTrain Genes with minimal occurences\n", minimal_occ_genes)
print("\nTest Genes with minimal occurences\n", minimal_occ_genes_test)

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_df[train_df["Class"]==((i*3+j)+1)].groupby('Gene')["ID"].count().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="ID", data=sorted_gene_group_top_7, ax=axs[i][j])

#text Count        
train_df.loc[:, 'Text_count']  = train_df["Text"].apply(lambda x: len(x.split()))
print(train_df.head())



