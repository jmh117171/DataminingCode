import numpy as np
import matplotlib.pyplot as plt
import math

#定义高斯函数
def Gaussian(x,mean,std):
    return np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))

if __name__=="__main__":
    #读取magic数据
    telescopeData = np.loadtxt("../datasets/magic04.data",float,delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9))
    print(telescopeData)

    #Compute the multivariate mean vector
    meanOfColumn = telescopeData.mean(axis=0)   #计算每列属性的均值，axis=0
    print(" the multivariate mean vector:")
    print(meanOfColumn)

    #Compute the sample covariance matrix as inner products between the columns of the centered data matrix.
    covarianceMatrix = []
    centeredPoints_attribute = telescopeData - meanOfColumn     #每列属性减去均值，完成点的中心化
    for i in range(len(centeredPoints_attribute[0])):
        temp_vector = []
        for j in range(len(centeredPoints_attribute[0])):
            #计算i和j列属性的内积
            temp_vector.append(np.dot(centeredPoints_attribute[:,i],centeredPoints_attribute[:,j]))
        covarianceMatrix.append(temp_vector)
    covarianceMatrix = np.array(covarianceMatrix)
    #得到协方差矩阵，这里用的是无偏的计算公式（除以n-1）
    covarianceMatrix = covarianceMatrix / (len(centeredPoints_attribute) - 1)
    # covarianceMatrix = np.cov(telescopeData.T)        #这是直接调用协方差矩阵计算接口
    print(("the sample covariance matrix:"))
    print(covarianceMatrix)

    #Compute the sample covariance matrix as outer product between the centered data points. have question
    centeredPoints = telescopeData - meanOfColumn   #每列属性减去均值，完成点的中心化
    covarianceMatrix_useCenteredPoints = []
    for i in range(len(centeredPoints)):
        if i == 0:
            #计算点中心化后的张量积（outer product）
            covarianceMatrix_useCenteredPoints = np.matmul(np.array(centeredPoints[i][:]).reshape(len(centeredPoints[i][:]),1),
                                                            np.array(centeredPoints[i][:]).reshape(1,len(centeredPoints[i][:])))
        else:
            covarianceMatrix_useCenteredPoints += np.matmul(np.array(centeredPoints[i][:]).reshape(len(centeredPoints[i][:]),1),
                                                            np.array(centeredPoints[i][:]).reshape(1,len(centeredPoints[i][:])))
    #同样用的是无偏的样本协方差计算公式
    covarianceMatrix_useCenteredPoints = covarianceMatrix_useCenteredPoints / (len(centeredPoints) - 1)
    print("use centered points outer product compute the covariance:")
    print(covarianceMatrix_useCenteredPoints)

    #Compute the correlation between Attributes 1 and 2
    print("attributes1 and attributes2 correlation:")
    attributes1 = telescopeData[:,0]    #获取到属性1
    attributes2 = telescopeData[:,1]    #获取到属性2
    # print(np.corrcoef([attributes1,attributes2]))     #调用相关系数矩阵计算接口
    mean1 = np.mean(attributes1)    #计算属性1的均值
    mean2 = np.mean(attributes2)    #计算属性2的均值
    #利用相关系数公式计算两个属性之间的相关性
    attr1AndAttr2Corrcoef = np.dot(attributes1-mean1,attributes2-mean2)/(np.dot(attributes1-mean1,attributes1-mean1)*np.dot(attributes2-mean2,attributes2-mean2))**0.5
    print(attr1AndAttr2Corrcoef)

    # Plot the scatter plot between these two attributes
    #利用scatter函数绘制属性1和属性2之间的散点图
    plt.scatter(attributes1, attributes2, s=10, marker='o')
    plt.xlabel("fLength(mm)")   #设置横坐标标题
    plt.ylabel("fWidth(mm)")    #设置纵坐标标题
    plt.title("the scatter between attribute1(fLength) and attribute2(fWidth)") #设置标题
    plt.show()  #展示散点图绘制结果

    #Assuming that Attribute 1 is normally distributed, plot its probability density function
    std1 = np.std(attributes1)      #样本方差（无偏的）除以n-1
    x1 = np.linspace(mean1 - 6 * std1, mean1 + 6 * std1, 100)   #设置自变量x的范围和划分间隔
    plt.plot(x1, Gaussian(x1,mean1,np.std(attributes1)), 'r')   #绘制高斯曲线
    plt.xlabel("attribute 1 of the fLength(mm)")    #设置横坐标标题
    plt.ylabel("probability density value")     #设置纵坐标标题
    plt.title("the normally distributed probability density function of attribute 1")   #设置标题
    plt.show()  #展示高斯曲线绘制结果

    #Which attribute has the largest variance, and which attribute has the smallest variance? Print these values.
    print("attributes variance vector:")
    varianceVector = np.var(telescopeData,axis=0)   #计算每列属性的方差
    print(varianceVector)
    print("the smallest variance and the the largest variance:")
    #找到方差最小的是哪个属性，并输出最小值
    print("the smallest variance attribute is "+ str(varianceVector.argmin()+1) + "th attribute "
          "and its value is " + str(min(varianceVector)))
    #找到方差最大的是哪个属性，并输出最小值
    print("the largest variance attribute is " + str(varianceVector.argmax()+1) + "th attribue "
          "and its value is " + str(max(varianceVector)))

    #Which pair of attributes has the largest covariance, and which pair of attributes has the smallest covariance? Print these values.
    # posMax = np.unravel_index(np.argmax(covarianceMatrix), covarianceMatrix.shape)
    # print(posMax)
    # posMin = np.unravel_index(np.argmin(covarianceMatrix), covarianceMatrix.shape)
    # print(posMin)
    maxIndex_row = 0
    maxIndex_col = 1
    minIndex_row = 0
    minIndex_col = 1
    for i in range(len(covarianceMatrix)):
        for j in range(len(covarianceMatrix[i])):
            #寻找协方差的最大最小值，所以不考虑对角线（对角线是方差）
            if (i == j):
                continue
            if (abs(covarianceMatrix[i][j]) > abs(covarianceMatrix[maxIndex_row][maxIndex_col])):
                maxIndex_row = i
                maxIndex_col = j
            if (abs(covarianceMatrix[i][j]) < abs(covarianceMatrix[minIndex_row][minIndex_col])):
                minIndex_row = i
                minIndex_col = j
    print("attribute " + str(maxIndex_row+1) + " and attribute " + str(maxIndex_col+1) + " have the largest covariance")
    print("attribute " + str(minIndex_row + 1) + " and attribute " + str(minIndex_col + 1) + " have the largest covariance")



