import numpy as np 
import xlrd
import time
import pandas as pd


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import pandas as pd


class CTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)

def Encode_mirna(items_file):
    chars = "ACGTN-"
    ctable = CTable(chars)
    MAXLEN = 30
    mirna_data = pd.read_csv(items_file)
    mirna = mirna_data.iloc[:,1]
    x = np.zeros((len(mirna), MAXLEN, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(mirna):
        sentence = sentence.upper()
        sentence = sentence.replace('U','T')
        if len(sentence)>30:
            sentence=sentence[0:30]
        elif len(sentence)<30:
            sentence = sentence+'-'*(30-len(sentence))
        x[i] = ctable.encode(sentence, MAXLEN)
#    mirna_ae = keras.models.load_model("Models\\rna_model_64.h5")
    from tensorflow.keras.models import load_model ###
    mirna_ae = load_model("Models\\rna_model_64.h5", compile = False)

    embed_mirna_ae= Model(inputs=mirna_ae.input, outputs=mirna_ae.get_layer('lstm_5').output)
    embed_vector = embed_mirna_ae.predict(x)
    header = ['mirID']+['f_'+str(f) for f in range(64)]
    output = []
    for s in range(len(mirna)):
        d = [mirna_data.iloc[s,0]]+ [m for m in embed_vector[s]]
        output.append(d)
    pd_res = pd.DataFrame(output, columns=header)
    pd_res.to_csv('Data\\mirna-encoding-64-id.csv',index=None)
    
    
def Encode_sm(items_file):
    chars = ['\\', "'", '=', 'E', '@', 'G', 'U', '/', 'D', 'A', 'Q', 'Z', ':', '5', 'O', 'C', '.', '2', ']', ')', 'N', 'R', '6', 'T', 'V', 'S', '*', 'F', 'B', 'X', 'Y', ';', 'W', '-', ' ', '7', '+', '1', '3', 'L', 'J', '%', '{', '&', '(', '^', 'M', '}', 'K', '9', '|', '0', '[', 'P', 'H', '8', 'I', '4', '#']
    ctable = CTable(chars)
    MAXLEN = 50
    sm_data = pd.read_csv(items_file)
    sm = sm_data.iloc[:,1]
    x = np.zeros((len(sm), MAXLEN, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sm):
        if len(sentence)>MAXLEN:
            sentence= sentence[0:MAXLEN]
        elif len(sentence)<MAXLEN:
            sentence =sentence +'^'*(MAXLEN-len(sentence))    
        sentence=sentence.upper()
        x[i] = ctable.encode(sentence, MAXLEN)
#    sm_ae = keras.models.load_model('Models\\sm_mmodel_64.h5')
    from tensorflow.keras.models import load_model ###
    sm_ae = load_model("Models\\sm_mmodel_64.h5", compile = False)
    embed_sm_ae= Model(inputs=sm_ae.input, outputs=sm_ae.get_layer('lstm').output)
    embed_vector = embed_sm_ae.predict(x)
    header = ['smID']+['f_'+str(f) for f in range(64)]
    output = []
    for s in range(len(sm)):
        d = [sm_data.iloc[s,0]]+ [m for m in embed_vector[s]]
        output.append(d)
    pd_res = pd.DataFrame(output, columns=header)
    pd_res.to_csv('Data\\sm_encoding_64_id.csv',index=None)
    
    
    

def filetomat(filename):
    f=open(filename,'r')
    array=f.readlines()
    lines=[]
    for line in array:
        line=line.strip('\n')
        linelist=[float(a) for a in line.split()]
        lines.append(linelist)
    matrix=np.array(lines)
    f.close()
    return matrix   

def ni0(a,i,j):
    ni0=a[i,j]
    return ni0
def ni1(a,i,j):
    ni1=(1-a[i,j])*(np.dot(a[i,:],a[j,:]))
    return ni1
def ni2(a,b,i,j):
    ni2=a[i,j]*(np.dot(a[i,:],b[j,:]))
    return ni2
def ni3(a,b,i,j):
    ni3=a[i,j]*(np.dot(b[i,:],a[j,:]))
    return ni3
def ni4(a,b,i,j):
    ni4=a[i,j]*(np.dot(a[i,:],a[j,:]))
    return ni4
  
def ni5(a,b,i,j):
    ni5=a[i,j]*np.dot((b[i,:]*a[j,:]),np.dot((b[i,:]*b[j,:]),a))
    return ni5
def ni6(a,b,i,j):
    ni6=(1-a[i,j])*np.dot((a[i,:]*a[j,:]),np.dot((b[i,:]*a[j,:]),b))
    return ni6
def ni7(a,b,i,j):
    ni7=(1-a[i,j])*np.dot((a[i,:]*b[j,:]),np.dot((b[i,:]*a[j,:]),a))
    return ni7
def ni8(a,b,i,j):
    ni8=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((b[i,:]*b[j,:]),a))
    return ni8
def ni9(a,b,i,j):
    ni9=(1-a[i,j])*np.dot((a[i,:]*b[j,:]),np.dot((a[i,:]*a[j,:]),b))
    return ni9
def ni10(a,b,i,j):
    ni10=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((b[i,:]*a[j,:]),b))
    return ni10
def ni11(a,b,i,j):
    ni11=(1-a[i,j])*np.dot((a[i,:]*a[j,:]),np.dot((b[i,:]*b[j,:]),a))
    return ni11
def ni12(a,b,i,j):
    ni12=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((a[i,:]*b[j,:]),b))
    return ni12
def ni13(a,b,i,j):
    ni13=a[i,j]*np.dot((b[i,:]*a[j,:]),np.dot((b[i,:]*a[j,:]),b))
    return ni13
def ni14(a,b,i,j):
    ni14=(1-a[i,j])*np.dot((a[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),b))
    return ni14
def ni15(a,b,i,j):
    ni15=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((b[i,:]*a[j,:]),a))
    return ni15
def ni16(a,b,i,j):
    ni16=a[i,j]*np.dot((b[i,:]*b[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni16
def ni17(a,b,i,j):
    ni17=a[i,j]*np.dot((a[i,:]*a[j,:]),np.dot((b[i,:]*b[j,:]),a))
    return ni17
def ni18(a,b,i,j):
    ni18=a[i,j]*np.dot((a[i,:]*a[j,:]),np.dot((b[i,:]*a[j,:]),b))
    return ni18
def ni19(a,b,i,j):
    ni19=(1-a[i,j])*np.dot((a[i,:]*b[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni19
def ni20(a,b,i,j):
    ni20=a[i,j]*np.dot((a[i,:]*a[j,:]),np.dot((a[i,:]*b[j,:]),b))
    return ni20
def ni21(a,b,i,j):
    ni21=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((a[i,:]*b[j,:]),a))
    return ni21
def ni22(a,b,i,j):
    ni22=(1-a[i,j])*np.dot((b[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni22
def ni23(a,b,i,j):
    ni23=(1-a[i,j])*np.dot((a[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni23
def ni24(a,b,i,j):
    ni24=a[i,j]*np.dot((a[i,:]*b[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni24
def ni25(a,b,i,j):
    ni25=a[i,j]*np.dot((a[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),b))
    return ni25
def ni26(a,b,i,j):
    ni26=a[i,j]*np.dot((b[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni26
def ni27(a,b,i,j):
    ni27=a[i,j]*np.dot((a[i,:]*a[j,:]),np.dot((a[i,:]*a[j,:]),a))
    return ni27


#def GraphletAssociationPrediction(pred_file,assoc_file, mir_sim, sm_sim, mir_no='mi_no.xlsx', sm_no='sm_no.xlsx'):
def GraphletAssociationPrediction(pred_file,assoc_file, mir_sim, sm_sim, mir_no, sm_no):

    ''' Calculate Graphlet interactions for miRNAs and Small Molecules
    @pred_file: Name of output prediction file:'.txt'
    @assoc_file: Name of input association file:'.txt'
    @mir_sim: Name of input miRNA similarity file:'.txt'
    @sm_sim: Name of input small molecule similarity file:'.txt'
    @mir_no: Name of input miRNA namne_to_no mapping file:'.xlsx'
    @sm_no: Name of input small molecule namne_to_no file:'.xlsx'
    '''
    print("Start: ", pred_file)
    print(time.strftime('%X %x'))
    fpredict=open(pred_file,'w')
    
    fnsm=xlrd.open_workbook(sm_no,'r')
    fnmi=xlrd.open_workbook(mir_no,'r')
            
  
    mirna_0=filetomat(mir_sim)  
    smallmo_0=filetomat(sm_sim)  
    assoarr_0=filetomat(assoc_file) 
    
    r1= mirna_0.shape[0] # no. of all imput miRNAs
    r2= smallmo_0.shape[0] # no. of all input SMs
    assoc = assoarr_0.shape[0] # o. of associations
    del mirna_0
    del smallmo_0
    del assoarr_0
    
    mirna=np.zeros([r1,r1])
    smallmo=np.zeros([r2,r2])    
    assoarr=np.zeros([assoc,2]) 
    print(r1,r2,assoc)
    mirna=filetomat(mir_sim)  
    smallmo=filetomat(sm_sim)  
    assoarr=filetomat(assoc_file) 
    
    

    
    adj=np.zeros((r2,r1))
    for i in range(assoarr.shape[0]):
        adj[(int(assoarr[i,0])-1),(int(assoarr[i,1])-1)]=1        
        
    sm=mirna.copy()
    sd=smallmo.copy()
              
    eye1=np.eye(r1,r1)
    eye2=np.eye(r2,r2)
    reye1=1-eye1
    reye2=1-eye2
    smv=reye1*(1-sm)
    sdv=reye2*(1-sd)
    
    GI=np.zeros([r1,r1,28])
    sumi=np.zeros([r1,28])
    
    for i in range(r1):
      for j in range(r1):
        if j!=i:
          GI[i,j,0]=ni0(sm,i,j)
          GI[i,j,1]=ni1(sm,i,j)
          GI[i,j,2]=ni2(sm,smv,i,j)
          GI[i,j,3]=ni3(sm,smv,i,j)
          GI[i,j,4]=ni4(sm,smv,i,j)
          GI[i,j,5]=ni5(sm,smv,i,j)
          GI[i,j,6]=ni6(sm,smv,i,j)
          GI[i,j,7]=ni7(sm,smv,i,j)
          GI[i,j,8]=ni8(sm,smv,i,j)
          GI[i,j,9]=ni9(sm,smv,i,j)
          GI[i,j,10]=ni10(sm,smv,i,j)
          GI[i,j,11]=ni11(sm,smv,i,j)
          GI[i,j,12]=ni12(sm,smv,i,j)
          GI[i,j,13]=ni13(sm,smv,i,j)
          GI[i,j,14]=ni14(sm,smv,i,j)
          GI[i,j,15]=ni15(sm,smv,i,j)
          GI[i,j,16]=ni16(sm,smv,i,j)
          GI[i,j,17]=ni17(sm,smv,i,j)
          GI[i,j,18]=ni18(sm,smv,i,j)
          GI[i,j,19]=ni19(sm,smv,i,j)
          GI[i,j,20]=ni20(sm,smv,i,j)
          GI[i,j,21]=ni21(sm,smv,i,j)
          GI[i,j,22]=ni22(sm,smv,i,j)
          GI[i,j,23]=ni23(sm,smv,i,j)
          GI[i,j,24]=ni24(sm,smv,i,j)
          GI[i,j,25]=ni25(sm,smv,i,j)
          GI[i,j,26]=ni26(sm,smv,i,j)
          GI[i,j,27]=ni27(sm,smv,i,j)
    sumi=GI.sum(axis=1)
    
    GID=np.zeros([r2,r2,28])
    sumid=np.zeros([r2,28])
    
    for i in range(r2):
      for j in range(r2):
        if j!=i:
          GID[i,j,0]=ni0(sd,i,j)
          GID[i,j,1]=ni1(sd,i,j)
          GID[i,j,2]=ni2(sd,sdv,i,j)
          GID[i,j,3]=ni3(sd,sdv,i,j)
          GID[i,j,4]=ni4(sd,sdv,i,j)
          GID[i,j,5]=ni5(sd,sdv,i,j)
          GID[i,j,6]=ni6(sd,sdv,i,j)
          GID[i,j,7]=ni7(sd,sdv,i,j)
          GID[i,j,8]=ni8(sd,sdv,i,j)
          GID[i,j,9]=ni9(sd,sdv,i,j)
          GID[i,j,10]=ni10(sd,sdv,i,j)
          GID[i,j,11]=ni11(sd,sdv,i,j)
          GID[i,j,12]=ni12(sd,sdv,i,j)
          GID[i,j,13]=ni13(sd,sdv,i,j)
          GID[i,j,14]=ni14(sd,sdv,i,j)
          GID[i,j,15]=ni15(sd,sdv,i,j)
          GID[i,j,16]=ni16(sd,sdv,i,j)
          GID[i,j,17]=ni17(sd,sdv,i,j)
          GID[i,j,18]=ni18(sd,sdv,i,j)
          GID[i,j,19]=ni19(sd,sdv,i,j)
          GID[i,j,20]=ni20(sd,sdv,i,j)
          GID[i,j,21]=ni21(sd,sdv,i,j)
          GID[i,j,22]=ni22(sd,sdv,i,j)
          GID[i,j,23]=ni23(sd,sdv,i,j)
          GID[i,j,24]=ni24(sd,sdv,i,j)
          GID[i,j,25]=ni25(sd,sdv,i,j)
          GID[i,j,26]=ni26(sd,sdv,i,j)
          GID[i,j,27]=ni27(sd,sdv,i,j)
    sumid=GID.sum(axis=1)
                            
    n=adj.sum(axis=1)                     
    nd=adj.sum(axis=0)                     
    
    xt=np.zeros([1,28])
    for i in range(r2):
      if int(n[i])>1:                         
        normmj=np.zeros([int(n[i]),28])       
        for mj in range(r1):               
          if adj[i,mj]==1:
            nmk=0                        
            for mk in range(r1):          
              if ((adj[i,mk]==1) and (mk!=mj)):
                for k in range(28):
                  if sumi[mk,k]!=0:
                    normmj[nmk,k]=(GI[mk,mj,k])/(sumi[mk,k])   
                nmk+=1
            xmj=normmj.sum(axis=0)        
            if (xmj.sum(0))!=0:           
              xt=np.vstack((xt,xmj))   
    
    np.delete(xt,0,0)
    x=xt.T
    v=np.zeros([28,1])
    s=np.ones([x.shape[1],1])
    
    xx=np.dot(x,xt)
    xx=np.mat(xx)
    xxi=np.linalg.pinv(xx)
    xxix=np.dot(xxi,x)
    v=np.dot(xxix,s)                      
    
    xtd=np.zeros([1,28])
    for j in range(r1):
      if int(nd[j])>1:                         
        normdi=np.zeros([int(nd[j]),28])       
        for di in range(r2):               
          if adj[di,j]==1:
            ndk=0                         
            for dk in range(r2):          
              if ((adj[dk,j]==1) and (dk!=di)):
                for k in range(28):
                  if sumid[dk,k]!=0:
                    normdi[ndk,k]=(GID[dk,di,k])/(sumid[dk,k])   
                ndk+=1
            xdi=normdi.sum(axis=0)       
            if (xdi.sum(0))!=0:          
              xtd=np.vstack((xtd,xdi))   
    
    np.delete(xtd,0,0)
    xd=xtd.T
    vd=np.zeros([28,1])
    sd=np.ones([xd.shape[1],1])
    
    xxd=np.dot(xd,xtd)
    xxd=np.mat(xxd)
    xxid=np.linalg.pinv(xxd)
    xxixd=np.dot(xxid,xd)
    vd=np.dot(xxixd,sd)                      
    
    
    st=np.zeros([r2,r1])
    stm=np.zeros([r2,r1])
    std=np.zeros([r2,r1])
    lists=[]
    for i in range(r2):
      xmj=np.array([28])
      normmj=np.zeros([int(n[i]),28])
      for mj in range(r1):
        normdi=np.zeros([int(nd[mj]),28])
        if adj[i,mj]==0:                   
          if int(n[i])!=0:                      
            nmk=0
            for mk in range(r1):
              if adj[i,mk]==1:
                for k in range(28):
                  if sumi[mk,k]!=0:
                    normmj[nmk,k]=(GI[mk,mj,k])/(sumi[mk,k])
                nmk+=1
            xmj=normmj.sum(axis=0)    
            stm[i,mj]=np.dot(xmj,v)    #score1
    
          if int(nd[mj])!=0:
            ndk=0
            for dk in range(r2):
              if adj[dk,mj]==1:
                for k in range(28):
                  if sumid[dk,k]!=0:
                    normdi[ndk,k]=(GID[dk,i,k])/(sumid[dk,k])
                ndk+=1
            xdi=normdi.sum(axis=0)    
            std[i,mj]=np.dot(xdi,vd)    #score2
    
          st[i,mj]=(stm[i,mj]+std[i,mj])/2
    
          listst=[st[i,mj],i,mj]
          lists.append(listst)          
    
    
    
    lists.sort(reverse=True)            
    table1=fnsm.sheet_by_name('sm') 
    table2=fnmi.sheet_by_name('mirna')  
    for i in range(len(lists)):         
      fpredict.writelines([str(table1.cell(lists[i][1],0).value),'\t',str(table2.cell(lists[i][2],0).value),'\t',str(lists[i][0]),'\n'])        
    fpredict.close()
    print("Finnished")
    print("End: ", pred_file)
    print(time.strftime('%X %x'))



def append_add(filename):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + ['add'])


def CheckSmilesExist(smiles,smiles_file):
    smiles_df = pd.read_csv(smiles_file,  sep=",")
    found_smiles = smiles_df.loc[smiles_df['SMILES'] == smiles]
#    found_smiles  = smiles_df['SMILES'].loc[smiles] 
    print(found_smiles)
    return found_smiles     
#    append_sm = smiles_df.append([[new_name,smiles]], ignore_index=True)
#    sm_smiles_new_f = 'smiles_new.csv'
#    append_sm.to_csv(sm_smiles_new, index=None, header=None)    
#    return sm_smiles_new_f

def CheckMirnasExist(mirnas,mirnas_file):
    mirnas_df = pd.read_csv(mirnas_file,  sep=",")
    found_mirnas = mirnas_df.loc[mirnas_df['mirnaSeq'] == mirnas]
#    found_smiles  = smiles_df['SMILES'].loc[smiles] 
    print(found_mirnas)
#    return len(found_mirnas) > 0
    return found_mirnas

def AppendRecordsAndEncode(new_items, input_type, seq_f, id_no_f):
    valid_type = {'sm','mir'}
    if input_type not in valid_type:
        raise ValueError("results: input_type must be one of %r." % valid_type)
        
    id_df = pd.read_excel(id_no_f, index_col=None, header=None)
    new_id  = int(id_df[1].iloc[-1]+1) # Add 1 to the id of the last record
    
    new_name = input_type+"_"+str(new_id)
    append_new_id = id_df.append([[new_name,new_id]], ignore_index=True)
    
    # Append new SMILES to SMILES file
    seq_df = pd.read_csv(seq_f,  sep=",")
    new_entries = {}
    sheet_name = ''
    if input_type == 'sm':
        new_entries = {'CID':[new_name],'SMILES':[new_items]}
        sheet_name = 'sm'
    elif input_type == 'mir':
        new_entries = {'miRNA':[new_name],'mirnaSeq':[new_items]}
        sheet_name = 'mirna'
    
    append_new_id.to_excel(id_no_f,sheet_name=sheet_name, index=None, header=None)   

    
    new_entries_df = pd.DataFrame(new_entries, index=None)
    append_sm = seq_df.append(new_entries_df, ignore_index=True)
    append_sm.to_csv(seq_f, index=None)
    
    # Write new entries in a separate file to be encoded
    new_entries_added_f = append_add(seq_f)
    new_entries_df.to_csv(new_entries_added_f, index=None)    
    new_entries_encode_f = AutoEncodeItem(new_entries_added_f, input_type) # Encoding here: 'sm': small molecules, 'mir': mirna seq.
    
    return new_entries_encode_f, new_name


from sklearn import preprocessing    
def NormAndNegate(df): # Normalize values, then negate to represent simmilarity
    #df = pd.read_csv(fname, sep="\t", header=None)
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)    
    for i, row in df.iterrows():
        for c in range(len(df.columns)):
            df.at[i,c]= 1- row[c]
    return df

from sklearn.metrics.pairwise import euclidean_distances
def ECDist(df):    
    #df_sm_831_2 = pd.read_csv('/home/ibrahim/GISMA/sm_encoding-id.csv')
#    print(df)
    enc_df = df#pd.read_csv(fname)
    enc_lst = enc_df.drop(enc_df.loc[:, :'f_0'],axis=1).values.tolist()
    dist = euclidean_distances(enc_lst, enc_lst)
#    print(enc_df)

#    id_list = enc_df[id_col_name].tolist()
    id_list = enc_df.iloc[:,0].tolist()
    print(enc_df)
    euc_dist_df = pd.DataFrame(dist, columns =id_list)  
    euc_dist_df.to_csv('ec_dist.txt', sep="\t", header=None, index=False)
    neg_df = NormAndNegate(euc_dist_df)
    return neg_df

## Please add code for encoding in this function according to type
## The items file contains items to be encoded, input_type indicates if it is sm or mirna
## The function should return a file name containing the encodings of the inputs in the items_file
def AutoEncodeItem(items_file, input_type): # smiles strings in csv file
#        sm_new_encode = smiles AutoEncoder(smiles_file)
    new_encode_f = ''
    if input_type == 'sm': # Add Code to encode small molecules in the items_file
        Encode_sm(items_file)
        new_encode_f= 'Data\\sm_encoding_64_id.csv' # Encoding of new smiles here, CSV file
    elif input_type == 'mir':  # Add Code to encode miRNAs in the items_file
        Encode_mirna(items_file)
        new_encode_f= 'Data\\mirna-encoding-64-id.csv' # Encoding of new mirnas here, CSV file
    return new_encode_f
    
def AppendEncodingsAndGetSMSim(sm_new_encode_f, sm_orig_encode_f, sim_f):
    sm_encode = pd.read_csv(sm_orig_encode_f,  sep=",")
    sm_new_encode_df = pd.read_csv(sm_new_encode_f,  sep=",")
    append_sm_encode = sm_encode.append(sm_new_encode_df, ignore_index=True)       
    append_sm_encode.to_csv('Data\\sm-temp.csv',index=False)

    new_sm_sim_df = ECDist(append_sm_encode)
    new_sm_sim_df.to_csv(sim_f, sep="\t", header=None, index=False)
    return sim_f

def AppendEncodingsAndGetMIRSim(mir_new_encode_f, mir_orig_encode_f, sim_f):
    mir_encode = pd.read_csv(mir_orig_encode_f,  sep=",")
    mir_new_encode_df = pd.read_csv(mir_new_encode_f,  sep=",")
    append_mir_encode = mir_encode.append(mir_new_encode_df, ignore_index=True)       
#    print('mir_orig_encode_f',mir_encode)
#    print('mir_new_encode_df',mir_new_encode_df)
#    print('append_mir_encode',append_mir_encode)
    append_mir_encode.to_csv('Data\\mir-temp.csv',index=False)

    new_mir_sim_df = ECDist(append_mir_encode)

#    new_mir_sim_df = ECDist(append_mir_encode)
    new_mir_sim_df.to_csv(sim_f, sep="\t", header=None, index=False)
    return sim_f

def GetInputPredictions(input_id, input_type, pred_file, pred_num=10):
    df = pd.read_csv(pred_file, sep="\t", header=None)
    if input_type == 'sm':
        pred_df = df.loc[df[0] == input_id]
        pred_lst = pred_df[1].tolist() # mirIDs ordered list
        
    elif input_type == 'mir':
        pred_df = df.loc[df[1] == input_id]
        pred_lst = pred_df[0].tolist() # SMIds ordered list    
    
    pred_dict_out=[]
    pred_val = pred_df[2].tolist() # list of prediction scores
    norm_pred_val = [(float(i)-min(pred_val))/(max(pred_val)-min(pred_val)) for i in pred_val] # normalize scores to be representative
    for num, name in enumerate(pred_lst): # create ordered list of tuples (id, score)
        pred_dict_out.append((name, round(norm_pred_val[num],3)))
    print(pred_dict_out[:pred_num]) 
    return pred_dict_out[:pred_num]    
    
    
#def CheckNewAssociations(new_smiles, new_mirs, pred,assoc, mir_sim, sm_sim,mir_no, sm_no):   
def CheckNewAssociations(new_smiles, new_mirs, file_names): 
    
    new_mir_sim = file_names['mir_sim']
    new_mir_no = file_names['mir_no']

    new_sm_sim = file_names['sm_sim']
    new_sm_no = file_names['sm_no']      
    new_sm_id=''
    new_mir_id = ''
    if len(new_mirs)>0:
        new_mir_encodes, new_mir_id = AppendRecordsAndEncode(new_mirs,'mir',file_names['mir_seq'], file_names['mir_no'])
        new_mir_sim = AppendEncodingsAndGetMIRSim(new_mir_encodes, file_names['mir_enc'], file_names['mir_sim'])
        print(new_mir_sim)

#    AppendRecordsAndEncode(new_smiles, seq_f='new_sm-smiles-out-831.csv', id_no_f='new_sm_no.xlsx', input_type)
    if len(new_smiles)>0:
        new_sm_encodes, new_sm_id  = AppendRecordsAndEncode(new_smiles,'sm', file_names['sm_smiles'], file_names['sm_no'])   
        new_sm_sim = AppendEncodingsAndGetSMSim(new_sm_encodes, file_names['sm_enc'], file_names['sm_sim'])
    GraphletAssociationPrediction(file_names['Pred_file'],file_names['known_assoc'], new_mir_sim, new_sm_sim,new_mir_no, new_sm_no)    
    if new_sm_id != '':
        GetInputPredictions(new_sm_id, 'sm', file_names['Pred_file'], pred_num=10)
    if new_mir_id != '':
        GetInputPredictions(new_mir_id, 'mir', file_names['Pred_file'], pred_num=10)


    

#GraphletAssiciationPrediction('prediction-584-sm128mir128-cos.txt', 'sm_mir_unq_assoc_584-540.txt', 
#                              'mir_128_similar_Cos-norm.txt','sm3_128_similar_Cos-norm.txt',  'mi_no.xlsx','sm_no.xlsx')
#GraphletAssiciationPrediction('prediction-no_sm139-sm128mir128.txt', 'assoc_no_sm139.txt', 
#                              'mir_128_similar_norm.txt','sm_128_similar_norm.txt',  'mi_no.xlsx','sm_no.xlsx')

### Add Input files names here
default_files = {'known_assoc':'Data\\sm_mir_unq_assoc_584.txt', # Experimentally known associations file
                 'Pred_file':'Data\\sm_mir_assoc_prediction.txt', # Output Precitions file
                 'sm_no':'Data\\new_sm_no.xlsx',                  # Updated name-no pair for SMs
                 'mir_no':'Data\\new_mi_no.xlsx',                 # Updated name-no pair for miRNAs
                 'sm_smiles':'Data\\sm-smiles.csv',               # List of SMILES
                 'mir_seq':'Data\\mirna-seq.csv',                 # List of miRNA sequqnces
                 'sm_enc':'Data\\sm_encoding_64.csv',             # Encodings for SMs
                 'mir_enc':'Data\\mir_encoding_64.csv',           # Encodings for miRNAs
                 'sm_sim':'Data\\sm_64_similar.txt',              # SMs similarities
                 'mir_sim':'Data\\mir_64_similar.txt'             # miRNAs similarities                 
                    }

# Main Code to run and test:
new_smiles = 'C1=CC(=C(C=C1CCN)O)OCC1CCOOCCCCC'
exist_smiles = CheckSmilesExist(new_smiles, default_files['sm_smiles'])

new_mirnas = 'UGAGGUAGUAGUUUGUACAGUUAU'
exist_mirnas = CheckMirnasExist(new_mirnas, default_files['mir_seq'])

if  len(exist_smiles) > 0:
    exist_smID = exist_smiles.iloc[0]['CID']
    print("SM File Not Updated, SMILES of: " + exist_smID +" already exist!")
#    GetInputPredictions(exist_smID,'sm',default_files['Pred_file'])
    new_smiles = []
    
if len(exist_mirnas) > 0:
    exist_mirID = exist_mirnas.iloc[0]['miRNA']
    print("MiRNA File Not Updated, Seq. of: " + exist_mirID +" already exist!")
    GetInputPredictions(exist_mirID,'mir',default_files['Pred_file'])
    new_mirnas = []

input_type = 'sm'
input_id = 'CID:34698'
pred_file = 'Data\prediction-584-old.txt' # default_files['Pred_file']

if  len(exist_smiles) == 0 or len(exist_mirnas) == 0:
    print('ReRun Needed')
    CheckNewAssociations(new_smiles, new_mirnas, default_files)
#    input_type = 'sm'
#    input_id = 'CID:34698'
#    pred_file = 'prediction-584-old.txt' # default_files['Pred_file']
#    GetInputPredictions(input_id, input_type, pred_file)
#    CheckNewAssociations(new_smiles, new_mirnas, 'prediction-584-sm128mir128-cos.txt', 
#                                     'sm_mir_unq_assoc_584-540.txt',
#                                     'mir_64_similar_norm.txt',
#                                     'sm3_64_similar_norm.txt' ,
#                                     'new_mi_no.xlsx','new_sm_no.xlsx')
##    
    

    