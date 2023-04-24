from random import sample
import numpy as np
#讀取Data
def readData(fileName):
    with open(fileName, "r") as file1:
        read_in = file1.read()
          
    temp = read_in.split("\n")
    data = []
    for x in temp:
        data.append(list(x.split(",")))
        
    data.pop(0)    
    data.pop(-1)

    for x in range(len(data)):
        data[x] = data[x][1:]
            
    for x in range(len(data)):
        for y in range(len(data[0])):
            if data[x][y] == '':
                data[x][y] = None
            else :
                data[x][y] = float(data[x][y])
    # for x in range(len(data)):
    #     data[x] = data[x][:1000]
    return data


#讀取Label
def readLabel(fileName):
    with open(fileName, "r") as file1:
        read_in = file1.read()  
        
    data = read_in.split("\n")
    data.pop(-1)
    for x in range(len(data)):
        data[x] = data[x].split(',')[1]
    data = data[1:]
    
    return data           

#補缺失值
def missingValue(trainData,testData):
    average = []
    missingNum = []
    for x in range(len(trainData[0])):
        average.append(0)
        missingNum.append(0)
        
    for x in range(len(trainData)):
        for y in range(len(trainData[0])):
            #計算缺失值數量
            if trainData[x][y] == None:
                missingNum[y] += 1
            
            #加總數值
            else:
                average[y] += trainData[x][y]
    #算出各個特徵的平均值
    for x in range(len(trainData[0])):
        average[x] = average[x] / (len(trainData)-missingNum[x]) 
    
    #補trainData的缺失值
    for x in range(len(trainData)):
        for y in range(len(trainData[0])):
            if trainData[x][y] == None:
                trainData[x][y] = average[y]  
             
    #補testData的缺失值
    for x in range(len(testData)):
        for y in range(len(testData[0])):
            if testData[x][y] == None:
                testData[x][y] = average[y]  
            
                    
    return trainData, testData, average 

# def outlier(data,label,average):
#     Q1 = []
#     Q3 = []
#     IQR = []
#     for x in range(len(data[0])):
#         Q1.append(0)
#         Q3.append(0)
#         IQR.append(0)
        
#     for x in range(len(data[0])):
#         l = sorted(list((y[x] for y in data)))
#         Q3.append(l[int(len(data) * 0.75) - 1])
#         Q1.append(l[int(len(data) * 0.25) - 1])
#         IQR.append((Q3[x] - Q1[x]) * 1.5)
        
#     for x in reversed(range(len(data))):
#         for y in range(len(data[0])):
#             if data[x][y] > (Q3[y] + IQR[y]) or data[x][y] < (Q1[y] - IQR[y]):
#                 data[x][y] = average[y]
    
#     return data, label

#標準化
def normalize(trainData, testData, average):
    std = []
    
    for x in range(len(trainData[0])):
        std.append(0)
    
    #加總各項數據與平均的差值    
    for x in range(len(trainData)):
        for y in range(len(trainData[0])):
            std[y] += ((trainData[x][y] - average[y]) ** 2)
            
    #計算標準差        
    for x in range(len(trainData[0])):
        std[x] /= len(trainData)
        std[x] = std[x] ** 0.5
        
    #標準化    
    for x in range(len(trainData)):
        for y in range(len(trainData[0])):
            if std[y] != 0:
                trainData[x][y] = (trainData[x][y] - average[y]) / std[y]
    #若某特徵標準差為0 刪除該特徵
    for x in reversed(range(len(trainData[0]))):
        if std[x] == 0:
            for y in reversed(range(len(trainData))):
                del trainData[y][x]
                
    for x in range(len(testData)):
        for y in range(len(testData[0])):
            if std[y] != 0:
                testData[x][y] = (testData[x][y] - average[y]) / std[y]
    
    for x in reversed(range(len(testData[0]))):
        if std[x] == 0:
            for y in reversed(range(len(testData))):
                del testData[y][x]
                
    return trainData, testData

def knn(trainData, trainLabel, t, k):
    #testData和所有trainData的距離
    all_dis = [np.sum((np.array(train) - np.array(t)) ** 2) ** 0.5 for train in trainData]
    #將index依照距離小到大排序
    idx = sorted(np.arange(len(all_dis)), key=lambda i: all_dis[i])
    #將距離排序
    all_dis = sorted(all_dis)
    #取前k筆資料
    result = np.array([all_dis[:k], idx[:k]]).T
    
    outcome = [0, 0, 0]
    #和最近的資料距離為0,代表為相同資料,直接回傳結果
    if result[0][0] == 0:
        return trainLabel[result[0][1]]
    #使用距離反比來加權,並將數值放大以利觀察
    for x in range(k):
        if trainLabel[result[x][1].astype('int32')] == 'BRCA':
            outcome[0] += ((1 / all_dis[x]) * 1000) ** 2
        elif trainLabel[result[x][1].astype('int32')] == 'KIRC':
            outcome[1] += ((1 / all_dis[x]) * 1000) ** 2
        elif trainLabel[result[x][1].astype('int32')] == 'LUAD':
            outcome[2] += ((1 / all_dis[x]) * 1000) ** 2
    
    #若大於一個定值,則代表該資料很有可能在某一已知的類別中
    if max(outcome) > (160 * k / 5):
        return (['BRCA', 'KIRC', 'LUAD'][outcome.index(max(outcome))])    
    #回傳0,代表此testData被判定為unknown
    return 0

#分群,原本使用k-medoids,但計算時間太久,故改用k-means        
def kmeans(testData, testLabel, predict):
    #將unknown的資料挑出來
    unknown = [x for x in range(len(predict)) if predict[x] == 0]
    centroid_dis = 0
    #先隨機選擇兩個形心
    initial = sample(unknown, 2)
    centroid = [testData[x] for x in range(len(testData)) if x in initial]
    #一開始選的形心距離太近或太遠,就重新選擇形心 
    while True:
        centroid_dis = 0
        #計算兩個形心的距離
        for x in range(len(testData[0])):
            centroid_dis += ((centroid[0][x] - centroid[1][x]) ** 2)
        #若距離太近或太遠會導致分群結果容易錯誤
        if  centroid_dis < 50000 or centroid_dis > 80000:   
            initial = sample(unknown, 2)
            centroid = [testData[x] for x in range(len(testData)) if x in initial]
        else:
            break
        
    clusters = [[],[]]
    clusters1 = [[],[]]
    count = 0
    #分群結果不再變動或分群50次後結束分群
    while count < 50:
        #將unknown的資料分到距離較近的那一類 
        for x in range(len(unknown)):
            distance = []
            for y in range(2):
                dis = np.sum((np.array(testData[unknown[x]]) - np.array(centroid[y])) ** 2)
                distance.append(dis ** 0.5)
            clusters[distance.index(min(distance))].append(unknown[x])
        
        #如果連續兩次分群結果相同,則結束分群
        if sorted(clusters1) == sorted(clusters):
            break;
        clusters1 = [[],[]]
        #儲存分群結果
        for x in range(len(clusters[0])):
            clusters1[0].append(clusters[0][x])
        for x in range(len(clusters[1])):
            clusters1[1].append(clusters[1][x])
        
        #計算新的形心位置
        for x in range(2):
            all_dis = []
            for y in range(len(testData[0])):
                all_dis.append(0)
            for z in range(len(clusters[x])):
                for k in range(len(testData[0])):
                    all_dis[k] += testData[clusters[x][z]][k]
            for n in range(len(all_dis)):
                all_dis[n] /= len(clusters[x])
            centroid[x] = all_dis
            
        clusters = [[],[]] 
        count += 1
    
    #預測結果
    for x in range(2):
        label = [0,0]
        for y in clusters[x]:
            if testLabel[y] == 'COAD':
                label[0] += 1
            elif testLabel[y] == 'PRAD':
                label[1] += 1
        number = label.index(max(label))
        for z in clusters[x]:
            predict[z] = ['COAD', 'PRAD'][label.index(max(label))]
      
    return predict, count + 1
                        
trainData = readData("train_data.csv")
trainLabel = readLabel("train_label.csv")
testData = readData("test_data.csv")
testLabel = readLabel("test_label.csv")
trainData, testData, average = missingValue(trainData, testData)
#trainData,trainLabel = outlier(trainData,trainLabel,average)
trainData, testData = normalize(trainData, testData, average)

print("")
print("k            accuracy        分群次數")

#讓knn的k值從1跑到21,並觀察準確率變化
for k in range(1,21):
    predict = []
    accuracy = 0
    for x in range(len(testData)):
        pred = knn(trainData, trainLabel, testData[x], k)
        predict.append(pred) 
          
    predict, count = kmeans(testData, testLabel,predict)
    #計算準確率
    for x in range(len(predict)):
        accuracy += (predict[x] == testLabel[x])
        
    accuracy /= len(testLabel) 
    print(k, accuracy, count, sep="\t")  
    
       

#分群預測完後將結果加入分類器中
for x in range(len(testData)):
    trainData.append(testData[x])
    trainLabel.append(predict[x])