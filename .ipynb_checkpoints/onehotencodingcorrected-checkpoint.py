#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np


# In[95]:


# Load the promoter and non-promoter sequences from files
promoter_sequences = []
with open('hs_promoter_TA','r') as f:
    seq = ''
    for line in f:
        if line.startswith('>'):
            if seq:
                promoter_sequences.append(seq)
                seq = ''
        else:
            seq += line.strip()
    promoter_sequences.append(seq)


# In[96]:


non_promoter_sequences = []
with open('hs_nonpromoter_TA','r') as f:
    seq = ''
    for line in f:
        if line.startswith('>'):
            if seq:
                non_promoter_sequences.append(seq)
                seq = ''
        else:
            seq += line.strip()
    non_promoter_sequences.append(seq)


# In[97]:


# Convert the sequences to one-hot encoding
def seq_to_one_hot(seq):
    ALPHABET = ['A', 'C', 'G', 'T']
    encoding = np.zeros((len(ALPHABET), len(seq)), dtype=np.float32)
    #print('SUBSTRING : ',seq[:SEQ_LENGTH])
    for i, nucleotide in enumerate(seq[:len(seq)]):
        #print('i : ',i,'    nucleotide : ',nucleotide)
        if nucleotide not in ALPHABET:
            encoding[ALPHABET.index('A'), i] = 0.25
            encoding[ALPHABET.index('C'), i] = 0.25
            encoding[ALPHABET.index('G'), i] = 0.25
            encoding[ALPHABET.index('T'), i] = 0.25
        else:
            encoding[ALPHABET.index(nucleotide), i] = 1.0
    #print('\n')
    return encoding


# In[98]:


# Convert the sequences to one-hot encoding
def seq_to_one_hot_backup(seq):
    encoding = np.zeros((SEQ_LENGTH, len(ALPHABET)), dtype=np.float32)
    #print('SUBSTRING : ',seq[:SEQ_LENGTH])
    for i, nucleotide in enumerate(seq[:SEQ_LENGTH]):
        #print('i : ',i,'    nucleotide : ',nucleotide)
        if nucleotide == 'N':
            encoding[0, ALPHABET.index('A')] = 0.25
            encoding[1, ALPHABET.index('C')] = 0.25
            encoding[2, ALPHABET.index('G')] = 0.25
            encoding[3, ALPHABET.index('T')] = 0.25
        else:
            encoding[i, ALPHABET.index(nucleotide)] = 1.0
    #print('\n')
    return encoding


# In[99]:


promoter_sequences = np.array([seq_to_one_hot(seq) for seq in promoter_sequences])
non_promoter_sequences = np.array([seq_to_one_hot(seq) for seq in non_promoter_sequences])

# Create the training, validation, and test datasets
X_train_val = np.concatenate([promoter_sequences, non_promoter_sequences])
y_train_val = np.concatenate([[[1,0] for i in range(len(promoter_sequences))], [[0,1] for i in range(len(non_promoter_sequences))]])

# Split the data into training, validation, and test sets
indices = np.random.permutation(X_train_val.shape[0])
split1_index = int(X_train_val.shape[0] * 0.8)
split2_index = int(X_train_val.shape[0] * 0.9)
train_indices, val_indices, test_indices = indices[:split1_index], indices[split1_index:split2_index], indices[split2_index:]
X_train, y_train = X_train_val[train_indices], y_train_val[train_indices]
X_val, y_val = X_train_val[val_indices], y_train_val[val_indices]
X_test, y_test = X_train_val[test_indices], y_train_val[test_indices]


# In[115]:


# len(promoter_sequences)


# In[101]:


# len(promoter_sequences[0][0])


# In[102]:


# for x,i in enumerate(promoter_sequences):
#     if len(i)!=300:
#         print(x)


# In[103]:


# promoter_sequences = np.array([seq_to_one_hot(seq) for seq in promoter_sequences])


# In[104]:


# type(promoter_sequences[0][0][0])


# In[105]:


# promoter_sequences[0]


# In[106]:


# non_promoter_sequences[0]


# In[107]:


# X_train.save('X_train_enzyme.npy')
# y_train.save('y_train_enzyme.npy')
# X_val.save('X_val_enzyme.npy')
# y_val.save('y_val_enzyme.npy')
# X_test.save('X_test_enzyme.npy')
# y_test.save('y_test_enzyme.npy')


# In[108]:


with open('X_train_enzyme.npy', 'wb') as f:
    np.save(f, X_train)


# In[109]:


with open('y_train_enzyme.npy', 'wb') as f:
    np.save(f, y_train)


# In[110]:


with open('X_val_enzyme.npy', 'wb') as f:
    np.save(f, X_val)


# In[111]:


with open('y_val_enzyme.npy', 'wb') as f:
    np.save(f, y_val)


# In[112]:


with open('X_test_enzyme.npy', 'wb') as f:
    np.save(f, X_test)


# In[113]:


with open('y_test_enzyme.npy', 'wb') as f:
    np.save(f, y_test)


# In[ ]:





# In[ ]:





# In[92]:


abc = [[1,0] for i in range(len(promoter_sequences))]


# In[93]:


len(abc)


# In[ ]:




