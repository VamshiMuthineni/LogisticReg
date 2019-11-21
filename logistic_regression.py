
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt


def calculate_loss(train_data, thetas):
    X = train_data[:, 0:-1]
    X = X.T

    numerator = np.exp(np.dot(thetas.T, X))
    denom = 1.0 + np.sum(numerator, axis=0)
    probs = numerator / denom

    loss = 0.0
    Y = train_data[:, -1]
    for i in range(Y.shape[0]):
        actual_class = int(float(Y[i]))
        if actual_class == 4:
            loss += np.log(1.0 - np.sum(probs[:, i]))
        else:
            prob = probs[:, i]
            loss += np.log(prob[actual_class - 1])
    return loss * (-1.0 / Y.shape[0])


def stochastic_gradient_descent(train_data, val_data, n0, n1):
    m = config.batch_size
#     n0 = config.n0
#     n1 = config.n1
    max_epoch = config.max_epoch
    delta = config.delta
    k = config.k
    thetas = np.zeros((train_data.shape[1] - 1, k-1))
    thetas_copy = thetas
    losses = []
    losses.append(99999)    #some big value
    val_losses = []
    val_losses.append(99999)
    epochlist = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(1, max_epoch + 1):
        n = (1.0 * n0)/(n1 + epoch)
        np.random.shuffle(train_data)
        count = 0
        for batch in range(int(train_data.shape[0]/m)):
            batch_data = train_data[batch * m:(batch + 1) * m, :]
            X = batch_data[:, 0:-1]
            X = X.T
            Y = batch_data[:, -1]
            numerator = np.exp(np.dot(thetas.T, X))
            denom = 1.0 + np.sum(numerator, axis=0)

            probs = (1.0 * numerator) / denom
            indicators = []
            for each_y in range(len(Y)):
                classes = []
                actual_class = int(float(Y[each_y]))
                for each_class in range(1,config.k):
                    if each_class == actual_class:
                        classes.append(1)
                    else:
                        classes.append(0)
                indicators.append(classes)
            indicator_arr = np.array(indicators)
            # print("delta matrix ", indicator_arr.transpose())
            # print("train lables", Y)

            diff_terms = np.transpose(indicator_arr) - probs
            thetas_copy = []
            for diff_term in diff_terms:
                gradient = diff_term * X
                gradient_sum = np.sum(gradient, axis=1)
                thetas_copy.append(gradient_sum)
                #print("gradient sum ", gradient_sum)
            thetas_copy = thetas + (1.0 * n / Y.shape[0]) * np.array(thetas_copy).T
            thetas = thetas_copy

            # print("thetas after epoch", epoch, thetas)
        losses.append(calculate_loss(train_data, thetas_copy))
        val_losses.append(calculate_loss(val_data, thetas_copy))
        
        
        train_accuracy = findAccuracy(train_data, thetas_copy)
        train_accuracies.append(train_accuracy)
        val_accuracy = findAccuracy(val_data, thetas_copy)
        val_accuracies.append(val_accuracy)
    
        epochlist.append(epoch)
        if losses[epoch] < losses[epoch-1] and losses[epoch] > (1 - delta)*losses[epoch-1]:
            final_epoch = epoch
            print("Breaking at epoch:", epoch)
            break
        else:
            thetas = thetas_copy
        if epoch % 10 == 1:
            print("Loss at epoch ", epoch, "is :", losses[epoch])
        #print("theta at the end of epoch", thetas_copy)
    return thetas, epoch, losses, val_losses, epochlist, train_accuracies, val_accuracies


def findAccuracy(data, thetas):
    X = data[:,0:-1]
    X = X.T
    numerator = np.exp(np.dot(thetas.T, X))
    denom = 1.0 + np.sum(numerator, axis=0)
    probs = numerator * 1.0 / denom
    count = 0
    temp = []
    Y = data[:, -1]
    for i in range(Y.shape[0]):
        actual_class = int(float(Y[i]))
        prob = list(probs[:,i])
        prob.append(1.0 - sum(prob))
        index = np.argmax(prob)
        if index + 1 == actual_class:
            count += 1
            temp.append(index + 1)
    return count*1.0/Y.shape[0]


def read_feat_and_val_data(feature_file, label_file):
    x = pd.read_csv(feature_file, header=None)
    x_id = x.iloc[:, 0]
    x = x.iloc[:, 1:]
    x_normal = (x - x.mean(axis=0)) / x.std(axis=0)
    x_normal.insert(0, 0, x_id)
    x_normal["ones"] = 1
    y = pd.read_csv(label_file)
    y = y.rename(columns={"Id": 0, "Category": 1})
    x_y = pd.merge(x_normal, y, on=0, sort=True)
    x_y = x_y.iloc[:, 1:]
    return np.array(x_y)


def build_confusion_matrix(y_act, y_pred):
    print("len of actual y",len(y_act))
    print("len of preds of y", len(y_pred))
    conf_mat_array = np.zeros((4,4))
    print(conf_mat_array.shape)
    for y_label in range(len(y_act)):
        y_pred_label = int(float(y_pred[y_label]))-1
        #print(y_pred_label)
        y_act_label = int(float(y_act[y_label]))-1
        #print(y_act_label)
        conf_mat_array[y_act_label][y_pred_label] += 1
    return conf_mat_array


def report_conf_matrix(confusion_mat_df, title):
    plt.matshow(confusion_mat_df, cmap=plt.gray())
    plt.colorbar()
    ticks = np.arange(len(confusion_mat_df.columns))
    plt.xticks(ticks, confusion_mat_df.columns)
    plt.yticks(ticks, confusion_mat_df.index)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

def predictData(test_data_array, thetas):
    test_data = test_data_array.T
    numerator = np.exp(np.dot(thetas.T, test_data))
    denom = 1.0 + np.sum(numerator, axis=0)

    probs = (1.0 * numerator)/ denom

    predicted = []
    for i in range(test_data.shape[1]):
        prob = list(probs[:, i])
        prob.append(1.0 - sum(prob))
        index = np.argmax(prob)
        predicted.append(index + 1)
    return predicted

if __name__ == "__main__":

    #load training data along with lables.
    train_data = read_feat_and_val_data("/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/Train_Features.csv", "/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/Train_Labels.csv")

    #print(train_data.transpose())

    #load_validation_data()
    val_data = read_feat_and_val_data("/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/Val_Features.csv", "/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/Val_Labels.csv")

    merged_data = np.concatenate((train_data, val_data), axis=0)
    
    
    n0 = [0.1]
    n1 = [5]
    
    final_epochs = []
    for i in range(len(n0)):
        final_thetas, final_epoch, losses, val_losses, epochlist, train_accuracies, val_accuracies = stochastic_gradient_descent(train_data, val_data, n0[i], n1[i])
        final_epochs.append(final_epoch)
        
    print("faster convergence at ", final_epochs.index(min(final_epochs)))
    print("n0", n0[final_epochs.index(min(final_epochs))], " n1", n1[final_epochs.index(min(final_epochs))])


    final_thetas, final_epoch, losses, val_losses, epochlist, train_accuracies, val_accuracies = stochastic_gradient_descent(train_data, val_data, n0[final_epochs.index(min(final_epochs))], n1[final_epochs.index(min(final_epochs))])
    plt.plot(epochlist, losses[1:], label="train loss")
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('plot L(theta) as function of epoch for faster n0,n1')
    plt.legend()
    plt.show()


    final_thetas, final_epoch, losses, val_losses, epochlist, train_accuracies, val_accuracies = stochastic_gradient_descent(train_data, val_data, 0.1, 1)
    plt.plot(epochlist, losses[1:], label="train loss")
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('plot L(theta) as function of epoch')
    plt.legend()
    plt.show()
    
    plt.plot(epochlist, losses[1:], label="train loss")
    plt.plot(epochlist, val_losses[1:], label="validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('plot L(theta) as function of epoch, train data and validation data')
    plt.legend()
    plt.show()
    
    plt.plot(epochlist, train_accuracies, label="train accuracy")
    plt.plot(epochlist, val_accuracies, label="validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('plot accuracy as function of epoch, train data and validation data')
    plt.legend()
    plt.show()

    train_accuracy = findAccuracy(train_data, final_thetas)
    validation_accuracy = findAccuracy(val_data, final_thetas)
    merged_accuracy = findAccuracy(merged_data, final_thetas)

    print("Train Data Accuracy      :", train_accuracy)
    print("Validation Data Accuracy :", validation_accuracy)
    print("Merged Data Accuracy     :", merged_accuracy)

    #test_data, test_features_id = load_test_data()

    test_x = pd.read_csv("/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/Test_Features.csv", header=None)
    test_x_id = test_x.iloc[:, 0]
    test_x = test_x.iloc[:, 1:]
    test_x_squares = test_x ** 2
    #     test_features_normalized =  test_features / np.sqrt(test_features_squares.sum(axis = 0))
    test_x_normal = (test_x - test_x.mean(axis=0)) / test_x.std(axis=0)
    test_x_normal.insert(0, 0, test_x_id)
    test_x_normal["ones"] = 1
    test_data_array = np.array(test_x_normal.iloc[:, 1:])
    test_x_id = np.array(test_x_id)
    test_x_id = np.reshape(test_x_id, (test_x_id.shape[0], 1))


    #predict_on_test_data(test_data, test_features_id, final_thetas)

    test_data = test_data_array.T
    numerator = np.exp(np.dot(final_thetas.T, test_data))
    denom = 1.0 + np.sum(numerator, axis=0)

    probs = (1.0 * numerator)/ denom

    predicted = []
    for i in range(test_data.shape[1]):
        prob = list(probs[:, i])
        prob.append(1.0 - sum(prob))
        index = np.argmax(prob)
        predicted.append(index + 1)
    preds = pd.DataFrame(data=predicted, columns=["Category"])

    preds.insert(0, "Id", test_x_id)
    preds.to_csv("/Users/vamshimuthineni/Vamshi/ML/assign3/cse512hw3/output.csv", index=False)

    
    train_preds = predictData(train_data[:, :-1], final_thetas)
    train_confusion = build_confusion_matrix(train_data[:, -1], train_preds)
    train_confusion = train_confusion.astype(int)
    train_confusion_df = pd.DataFrame(train_confusion, index=np.array(range(1,5)), columns = [1,2,3,4])
    print(train_confusion_df)
    
    report_conf_matrix(train_confusion_df, title = 'Confusion Matrix for Train data')
    
    
    val_preds = predictData(val_data[:, :-1], final_thetas)
    val_confusion = build_confusion_matrix(val_data[:, -1], val_preds)
    val_confusion = val_confusion.astype(int)
    val_conf_df = pd.DataFrame(val_confusion, index=np.array(range(1,5)), columns = [1,2,3,4])
    print(val_conf_df)
    
    report_conf_matrix(val_conf_df, title = 'Confusion Matrix for Validation data')
    


    
    
    
    
