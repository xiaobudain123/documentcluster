# -*- coding: utf-8 -*-
import jieba
import csv
import os
import re
import collections
import pandas as pd
from os import listdir
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans as k_means
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import scipy
from 机器学习.jqmcvi import base
from scipy.spatial.distance import  cdist
from scipy.spatial.distance import euclidean
import warnings



class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.tfidtransformer = TfidfTransformer()
        """
            CountVectorizer()
                输入：文档 corpus
                输出：该类将会把文本中的词语转为词频矩阵，即文档中各个单词的词频TF（即每个单词在文档中出现的次数），矩阵元素a[i][j]，表示词i在文章j中出现的频率

            TfidfTransformer()
                输入：词频TF
                输出：该类将会把文本转为词频矩阵，并且计算TF-IDF，即统计每个词语的词频逆反文档频率TF-IDF（即词频TF与逆反文档频率IDF的乘积，IDF的标准计算公式为 ：idf=log[n/(1+df)]，其中n为文档总数，df为含有所计算单词的文档数量，df越小，idf值越大，也就是说出现频率越小的单词意义越大）

            因此，利用以上两个函数，要计算某个文档的TF-IDF，需要两步完成：
                1.计算词频TF，通过函数CountVectorizer()来完成，以该文档为输入，并得到词频 tf 输出；
                2.计算词频逆反文档频率TF-IDF，通过函数TfidfTransformer()来实现，以第一步的词频 tf 输出为输入，并得到 tf-idf 格式的输出。
        """

    def load_stopwords(self, stopwords=None):
        """
        加载停用词
        :param stopwords:
        :return:
        """
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path):
        """
        文本预处理，
        将当前目录corpus_path下的，每一个文件进行预处理，然后将每一个文本处理的结果作为一行，存储到另一个文件中
        :param corpus_path:
        :return:
        """
        docLabels = [f for f in listdir(corpus_path) if f.endswith('.txt')]  ## 将当前corpus_path需要进行文本分词的文件名 存到docLabels里面
        corpus = []
        for doc in docLabels:
            with open(corpus_path + "/" + doc, 'r', encoding='utf-8-sig') as f:
                sentence = ''
                # 先把一个文本里面的数据转变成一行，然后在进行分词
                for line in f.readlines():  # readlines()一次性读取所有文本数据，readline()是一行一行的读取
                    reg_html = re.compile(r'<[^>]+>', re.S)  # 去除的是网页符号
                    line = reg_html.sub('', line)
                    line = re.sub('\s', '', line)
                    line = re.sub(pattern=r"\d", repl=r"", string=line)  # 去除字符串中的数字
                    sentence += line
                data = ' '.join([word for word in jieba.lcut(sentence.strip(),cut_all=False) if
                                 word not in self.stopwords])  # ''.join()目的是把分词后的list集合变成一句话，sentence.strip()去除首尾空格
                # print('*****',doc,"***",data)
                corpus.append(data)
        # print(corpus)
        return corpus   #返回的是这种形式corpus =['课本1目录'，'课本2目录'，'课本3目录'，'课本4目录']

    def get_text_tfidf_matrix(self, corpus):
        """
        抽取TF-IDF矩阵
        :param corpus:输入的文本语料
        :return:TF-IDF矩阵
        """
        # 把语料文本转为词频矩阵即：vectorizer.fit_transform(corpus)，并且计算TF-IDF 即：tfidtransformer.fit_transform（）
        tfidf = self.tfidtransformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取词袋中所有词语
        # words = self.vectorizer.get_feature_names()

        # 将TF-IDF矩阵抽取出来，元素a[i][j]表示词j在i类文本中的TF-IDF权重
        weights = tfidf.toarray()
        return weights

    def kmeans(self,  corpus_path,  weights, n_clusters=6):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        # corpus = self.preprocess_data(corpus_path)  # 数据预处理
        # weights = self.get_text_tfidf_matrix(corpus)  # 获取TF—idf
        clf = KMeans(n_clusters=n_clusters,random_state=9)

        # print(clf.fit(weights))

        y = clf.fit_predict(weights)  # 返回的是每个样本所属于的聚类类别,形如[3 3 2 0 0 3 3 3 3 3 0 2 ....},  #clf.fit_predict()用训练器数据X拟合分类器模型并对训练器数据X进行预测
        print("最后聚类的所属的标签",y)

        # 中心点
        # centers = clf.cluster_centers_
        # print(centers)

        # 用来评估簇的个数是否合适,距离约小说明簇分得越好,选取临界点的簇的个数
        # score = clf.inertia_

        # 每个样本所属的簇
        result = {}
        # print(y)
        docLabels = [f for f in listdir(corpus_path) if f.endswith('.txt')]  # 将需要进行文本分词的文件名 存到docLabels里面
        for text_idx, label_idx in zip(docLabels, y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)

        # 保存的第一种方式
        # 将分类的结果存储到csv中, 这种情况以最长的列为看齐，其他短的，没有值的部分填充为Nan， 因此使用使用的时候需要注意进行判断
        dataframe = pd.DataFrame.from_dict(result, orient='index')
        # dataframe = dataframe.reset_index().rename(columns={'index': 'key'})
        dataframe.T.to_csv('G:/句子相似度计算/聚类结果新6-121.csv', index=False, sep=',')
        print("每个类簇里面有哪些课程：",result)

        # 聚类评价指标－DB指数   指数越低聚类的效果越好
        daviesbouldinscore=davies_bouldin_score(weights, y)
        print("聚类评价指标－DB指数",daviesbouldinscore)
        return y ,result


    #K值选取评价，第一种方法 手肘法评价K值
    def shouzhoufa(self,weights,start=1,end=10):
        SSE = []  # 存放每次结果的误差平方和
        K=range(start,end)
        for k in K:
            estimator = KMeans(n_clusters=k,random_state=9)  # 构造聚类器
            estimator.fit(weights)
            # SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
            SSE.append(
                sum(
                    np.min(
                        cdist(weights, estimator.cluster_centers_, metric='euclidean'), axis=1))
                / weights.shape[0])
        plt.xlabel('k值')
        plt.ylabel('SSE')
        plt.plot(K, SSE, 'o-')
        plt.show()


    #K值选取评价，第二种方法 轮廓系数法评价K值
    def lunkuoxishufa(self,weights,start=1,end=10):
        silhouette_Scores = []  # 存放轮廓系数,根据轮廓系数的计算公式，只有一个类簇时，轮廓系数为0
        K = range(start+1,end)
        for k in K:
            y_pred =KMeans(n_clusters=k,random_state=9).fit_predict(weights)
            silhouette_Scores.append(silhouette_score(weights, y_pred, metric='euclidean'))
        plt.xlabel('k')
        plt.ylabel('silhouette')
        plt.plot(K, silhouette_Scores, 'bx-')
        plt.title('the silhouette_Scores')
        plt.show()


        '*************************************************************************************'

    # 第三种方法 间隔统计量Gap Statistic  聚类K值的选择
    def gapstatistic(self, data, refs=None, nrefs=20, ks=range(1, 15)):
        """
        I: NumPy array, reference matrix, number of reference boxes, number of clusters to test
        O: Gaps NumPy array, Ks input list

        Give the list of k-values for which you want to compute the statistic in ks. By Gap Statistic
        from Tibshirani, Walther.
        """
        dst = euclidean
        k_means_args_dict = {
            'n_clusters': 0,
            # drastically saves convergence time
            'init': 'k-means++',
            'max_iter': 100,
            'n_init': 1,
            'verbose': False,
            # 'n_jobs':8
        }

        shape = data.shape

        if not refs:
            tops = data.max(axis=0)
            bottoms = data.min(axis=0)
            dists = scipy.matrix(scipy.diag(tops - bottoms))
            rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
            for i in range(nrefs):
                rands[:, :, i] = rands[:, :, i] * dists + bottoms
        else:
            rands = refs

        gaps = scipy.zeros((len(ks),))

        for (i, k) in enumerate(ks):
            k_means_args_dict['n_clusters'] = k
            kmeans = k_means(**k_means_args_dict, random_state=9)
            kmeans.fit(data)
            (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_

            disp = sum(
                [dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index], :]) for
                 current_row_index
                 in range(shape[0])])

            refdisps = scipy.zeros((rands.shape[2],))

            for j in range(rands.shape[2]):
                kmeans = k_means(**k_means_args_dict)
                kmeans.fit(rands[:, :, j])
                (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_
                refdisps[j] = sum(
                    [dst(rands[current_row_index, :, j], cluster_centers[point_labels[current_row_index], :]) for
                     current_row_index in range(shape[0])])

            # let k be the index of the array 'gaps'
            gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)

        gaps = gaps.tolist()
        K = gaps.index(max(gaps))
        print("Gap statict", gaps)
        print("K值", K+1)#数组的下标是从0开始的
        print('k所对应的K值 gap', max(gaps))
        plt.xlabel('k')
        plt.ylabel('Gap statict')
        plt.plot(ks, gaps, 'bx-')
        plt.title('the Gap statict_Scores')
        plt.show()


        return ks, gaps

    #purity纯度评价指标
    # def purity(self,result, label):
    #     # 计算纯度
    #
    #     total_num = len(label)
    #     cluster_counter = collections.Counter(result)
    #     original_counter = collections.Counter(label)
    #
    #     t = []
    #     for k in cluster_counter:
    #         p_k = []
    #         for j in original_counter:
    #             count = 0
    #             for i in range(len(result)):
    #                 if result[i] == k and label[i] == j:  # 求交集
    #                     count += 1
    #             p_k.append(count)
    #         temp_t = max(p_k)
    #         t.append(temp_t)
    #     purity_score = sum(t) / total_num
    #     print("聚类的评价指标purity",purity_score)
    #     return purity_score


    #去除信息警告
    warnings.filterwarnings("ignore")

    # 邓恩指数评价
    def dum(self,weights,n_clusters):
        y_pred = KMeans(n_clusters=n_clusters,random_state=9).fit_predict(weights)
        df = pd.DataFrame(weights)
        pred = pd.DataFrame(y_pred)
        pred.columns = ['Type']

        # 数据合并weight 和预测的值进行合并
        prediction = pd.concat([df, pred], axis=1)

        cluster_list = []
        for k in range(0,n_clusters):
            strname = 'clus' + str(k)
            strname = prediction.loc[prediction.Type == k]
            cluster_list.append(strname.values)

        print("邓恩指数dum", base.dunn(cluster_list))
        return base.dunn(cluster_list)


if __name__ == '__main__':

    #实例化聚类对象
    Kmeans = KmeansClustering(stopwords_path='G:/句子相似度计算/text_clustering-master/data/stop_words.txt')
    #需要处理数据的路径
    corpus_path ="G:/句子相似度计算/目录集-合并-没有去重/"
    #数据预处理
    corpus = Kmeans.preprocess_data(corpus_path)
    #将处理完的语料转为TF-IDF矩阵
    weights = Kmeans.get_text_tfidf_matrix(corpus)

    #文本聚类操作，并且将结果保存  返回的r是标签
    y ,result = Kmeans.kmeans(corpus_path=corpus_path, weights=weights,n_clusters=12)

    #第一种方法 手肘法， 聚类K值的选择
    Kmeans.shouzhoufa(weights=weights,end=15)

    # 第二种方法 手肘法， 聚类K值的选择
    Kmeans.lunkuoxishufa(weights=weights,end=15)

    #第三种方法 间隔统计量Gap Statistic  聚类K值的选择
    ks, gaps = Kmeans.gapstatistic(data=weights)

    #第四种方法 纯度（Purity） 对聚类进行评价，属于类外的评价
    # list(map(int,np.array(y)))
    # Kmeans.purity(result=list(map(int,np.array(result))),label=np.array(y))

    #第五种方法 dum邓恩指数 评价
    Kmeans.dum(weights,n_clusters=12)
