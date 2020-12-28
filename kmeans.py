import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def load_data(s,k):
    f = open('./data/'+s+'/image/' + str(k) + '.png','rb') 
    data = []
    img = image.open(f) #以列表形式返回图片像素值
    m,n = img.size #图片大小
    for i in range(m):
        for j in range(n):  #将每个像素点RGB颜色处理到0-1范围内并存放data
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n #以矩阵型式返回data，图片大小
def kmeans(s,k):
    img_data,row,col = load_data(s,k)
    label = KMeans(n_clusters=2).fit_predict(img_data)  #聚类中心的个数为2
    label = label.reshape([row,col])    #聚类获得每个像素所属的类别
    pic_new = image.new("L",(row,col))  #创建一张新的灰度图保存聚类后的结果
    
    #保证分类的标签一致
    if label[0][0] == 1:
        label = -(label-1)
        
    for i in range(row):    #根据所属类别向图片中添加灰度值
        for j in range(col):
            pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
    pic_new.save('./data/'+s+'/kmeans/' + str(k) + '.png')




for k in range('train',400):
    kmeans(k)

for k in range('test',20):
    kmeans(k)