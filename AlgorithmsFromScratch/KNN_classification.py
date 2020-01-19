'''Steps to implement KNN from scratch:
    
    1. Load data (training data + labels):
        
        i. load the entire data at once 
        ii. separate labels into another list and perform calculations on training set
    
    2. specify for what are you using the Algorithm :
        
        i. CLassification problem : Use Choice as mode (majority vote)
            
        ii. Regression problem : Use choice as mean (takes the mean of the top k training values)
            
    3. for each data sample:
        
        i. calculate the euclidean distance of each data point with the query sample
        ii. store the distance and the index of the data point
        
    4. Sort the list that has distance and index stored
    
    5. Store the top k values into another array
    
    6. for each element in array:
        
        pick the label of each element
        
    7. if choice == mode:
        
        count the number of 0 and 1 lables
        return the label with maximum votes


'''










from collections import Counter
import math


def knn(data, test, k, cal_distance_fn, choice):
    neighbor_distances_and_indexes = []
    for index,x in enumerate(data):
        cal_dist = cal_distance_fn(x[:-1],test)
        neighbor_distances_and_indexes.append((cal_dist,index))
    sorted_values = sorted(neighbor_distances_and_indexes)
    k_sorted_values = sorted_values[:k]
    k_nearest_labels = [data[i][1] for dist,i in k_sorted_values]
    return choice(k_nearest_labels)

def mode(labels):
    return Counter(labels).most_common(1)

def euclidean(point1, point2):
    squared_dist = 0
    for i in range(len(point1)):
        squared_dist += math.pow((point1[i] - point2[i]),2)
    return(squared_dist)


def main():
    # age of people col 0
    # likes /dislikes pineapple (class)
    clf_data = [
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]
    test_data = [26]
    #choice = mode specifies we want to do classification
    #k = 3 consider classes of top 3 values 
    #metric = use euclidean distance
    
    test_predict = knn(clf_data,test_data, k = 3, cal_distance_fn = euclidean, choice = mode)
    print(test_predict)
    
    
if __name__ == '__main__':
    main()
