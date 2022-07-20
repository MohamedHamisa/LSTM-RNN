#use data from kaggle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer #ميثود علشان يحول التيكست لارقام
from keras.preprocessing.text import Tokenizer #بيقسم الجمل او الكلمات ل توكنز و كل واحدة فيهم بتتحول ل فيكتور علي حسب معامل كل واحدة في صيغة الباينري او انتجر سيكونس
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D #بيعمل حذف للكونكشنز في دايمنشن كامل
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical #ميثود بتعملي الوان هوت انكودر
import re #ريجيولر اكسبريشن و ده موديول بستخدمه في ان ال بي علشان اقدر اعرف الاكسبريشن او الجملة دي مناسبة ل جملة تانية ولا لا و هي هي REGEX

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

data = data[data.sentiment != "Neutral"] #هنا علشان احذف الكومنتات او الداتا المتعادلة او الكومنتات اللي ملهاش رأي سوتء حلو او وحش
data['text'] = data['text'].apply(lambda x: x.lower()) #عاوز كل الحروف سمول علشان المويل مايتزاولش
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) #معناها اي اسبيشال كاركتر احذفه  
#لمادا فانشكن دي لما تعوز تعمل فانكشن لمدة صغيرة من الوقت او عاوز تعمل باس لفانكشن ك بارميتر ل فانكشن تانية

#علشان يعرف ايه البوزتيف و النيجتيف و يطبعه
print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)

#ايتاروز ميثود بتعمل ايتيراتور بيشتغل علي الصفوف في الداتا فريم
for idx,row in data.iterrows():  
    row[0] = row[0].replace('rt',' ') #امسحه لو بدأ ب كذا
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ') #هيقسمهم و يحذف الكلمات المتكررة
tokenizer.fit_on_texts(data['text'].values) #هيبدأ يطبق الميثود دي علي التيكست
X = tokenizer.texts_to_sequences(data['text'].values) #هيحول الدات لسكونس 
X = pad_sequences(X) #علشان يزود لو اقل من الماكس

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1])) #عدد الاعمدة من اكس
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#هبدأ اشتغل علي الواي علشان احولها لهوت وان انكودر
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
#لو عملت الراندم استيت ب رقم كل مرة هيطلعلك نفس الناتج لكن ل ماعملتوش او سيبته ب نن هيطلعلك كل مرة ناتج مختلف علشان نسبة التقسيم هتختلف
#تريننج
batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

#هيبدأ يقيس فيه كام واحدة صح

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1


print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

#هنا بجرب الكود

twt = ['Meetings: Because none of us is as dumb as all of us.']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")

