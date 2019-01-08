#include<iostream>
#include <math.h>
#include <vector>
#include <random>
#include <iomanip>
#include <Eigen/Core>
#include <algorithm>
using namespace std;
//高斯密度函数
float gussian_fx(float x){
    float weight=1/(sqrt(2*M_PI));
    float exow=-pow(x,2)/2;
    return (weight*exp(exow));
};
//求X<x 的概率
float guassian_pb(float x, float u,float sigma2){
    float detx=0.01;
    float pb=0;
    float MINX=-5;
    while (MINX<=(x-u)/sqrt(sigma2)){
        MINX+=detx;
        pb+=detx*gussian_fx(MINX);
    };
    return pb;
};
float guassian_pb_from_data(vector<float >data,float x){
    static int i=0,j=0;
    while(i<data.size()){
        if(data[i]<=x){i++;j++;}
        else{i++;};
    }
    return float(j)/data.size();
}
//最大似然估计，接受参数为向量，返回pair first为均值，second为方差
pair<float,float> MaxLikelihood(vector<float > &xi){
    float hatu=0; float hatsigma2=0;
    for (int i = 0; i <xi.size() ; ++i) {
        hatu+=xi[i];
    }
    for (int j = 0; j <xi.size() ; ++j) {
        hatsigma2+=pow(xi[j]-(hatu/xi.size()),2);
    }
    pair<float ,float > us={hatu/xi.size(),hatsigma2/xi.size()};
    return us;
};

//生成正态分布，mu均值，sigma2方差，默认个数500
vector<float >NormalGenerating(float mu,float sigma2,int length=500){
    random_device rd;
    std::mt19937 gen(rd());
    uniform_real_distribution<float > ud(0.0,1.0);
    int i=0;
    vector<float > seed;
    while(i<length) {
        float x = ud(gen);
        float y = ud(gen);
        float temp;
        if (i % 2) {
            temp =cos(2 * M_PI * x) * sqrt(-2 * log(1 - y));
        }
        else{
            temp=sin(2*M_PI*x)*sqrt(-2*log(1-y));
        }
        float ns=temp*sqrt(sigma2)+mu;
        i++;
        seed.push_back(ns);
    }
    return seed;
}
//生成正态分布，mu均值，sigma2方差，默认个数500
vector<float >NormalGenerating2(float mu,float sigma2,int length=500){
    random_device rd;
    std::mt19937 gen(rd());
    normal_distribution<float > normal(mu,sqrt(sigma2));
    vector<float >seed;
    while(length--){
        seed.push_back(normal(gen));
    }
    return seed;
}

//需要显示初始化
struct MultiNormal{
    int length=500;
    float u=0;
    float sigma2=0;
    vector<float> Gussiandata;
    MultiNormal(vector<float> data){
        u=MaxLikelihood(data).first;
        sigma2=MaxLikelihood(data).second;
        Gussiandata=data;
    }
};
//混合高斯分布，将多个正态分布相加
vector<float>MultiNormalAdding(vector <MultiNormal> &kargs){
    vector<float > reciver(500,0);
    int k=kargs.size();
    for (int loc = 0; loc <500 ; ++loc) {
        int j=0;
        while(j<k) {
            reciver[loc]+=kargs[j].Gussiandata[loc];
            j++;
        }
    }
    return reciver;
}
//混合高斯分布，将多个正态分布混合
vector<float>MultiNormalMerging(vector <MultiNormal> &kargs){
    int k=kargs.size();
    int Alllength=0;
    int i=0;
    while(i<k){
        Alllength+=kargs[i].Gussiandata.size();
        i++;
    }
    vector<float > reciver(Alllength,0);
    int j=0;
    int det=0;
    while(j<k){
        for (int l = 0; l <kargs[j].Gussiandata.size() ; ++l) {
            int pos=j-1;
            if(pos==-1)pos=0;
            reciver[l+j*kargs[pos].Gussiandata.size()]=kargs[j].Gussiandata[l];
        }
        j++;
    }
    return reciver;
}

//生成统计数据
vector<vector<float >> CalNormalData(vector<float > &multgassian,int quration,int GMM=2){
    vector<float > temp=multgassian;
    sort(temp.begin(),temp.end());
    float mindata=temp.front();
    float maxdata=temp.back();
    float middle=(mindata+maxdata)/2;
    cout<<middle<<endl;
    //cout<<temp.size()<<endl;
    //for(auto &item:temp)cout<<item<<endl;
    int qu=quration;//分区数
    float step=(maxdata-mindata)/qu;//分区数;
    vector<vector<float >> SortedHistData;
    int i=0;
    int pos=0;
    while(i<qu){
        vector<float> tempdata;
        while(mindata+i*step<=temp[pos]&&temp[pos]<=mindata+(i+1)*step){
            tempdata.push_back(temp[pos]);
            pos++;
        }
        if(tempdata.size()==0) tempdata.push_back(0);//防止该分区为空
        SortedHistData.push_back(tempdata);
        tempdata.clear();
        i++;
    }
    Eigen::MatrixXf QKI(quration,GMM);
    Eigen::MatrixXf SITA(GMM,2);//同列0表示mu 1表示sigma2
    SITA<<middle,middle,1,1;
    vector<float> cof;
    for (int i = 0; i <quration ; ++i) {
        for (int j = 0; j <GMM ; ++j) { //混合个数
            QKI(i,j)=guassian_pb(SortedHistData[i].back(),SITA(j,0),SITA(j,1))
                     -guassian_pb(SortedHistData[i].front(),SITA(j,0),SITA(j,1));
        }
    }
    for (int k = 0; k <quration ; ++k) {
        float temp=0;
        for (int l = 0; l <GMM ; ++l) {
            temp+=QKI(k,l);

        }
        cof.push_back(temp);
    }
    for (int m = 0; m <quration ; ++m) {
        for (int n = 0; n <GMM ; ++n) {
            if(!cof[m])QKI(m,n)=0;
            else QKI(m,n)=QKI(m,n)/cof[m];
        }

    }
    //cout<<QKI<<endl;



    return SortedHistData;

}

int main(){
    vector<float > seed1=NormalGenerating(0,2);
    vector<float > seed2=NormalGenerating2(10,2);
    //vector<float > seed3=NormalGenerating2(15,2);
    MultiNormal No1(seed1);
    MultiNormal No2(seed2);
   // MultiNormal No3(seed3);
    vector<MultiNormal> Multiseeds;
    Multiseeds.push_back(No1);
    Multiseeds.push_back(No2);
    //Multiseeds.push_back(No3);
    vector<float > merge=MultiNormalMerging(Multiseeds);
    //for(auto &item:merge)cout<<item<<endl;
    //cout<<"the mixed guassian mean is u="<<MaxLikelihood(merge).first<<endl<<"the mixed guassian variance is simga="<<MaxLikelihood(merge).second<<endl;
//    for(auto &item:CalNormalData(merge,20)){
//        for(auto &j:item){
//            cout<<j<<"  "<<endl;
//        }
//        cout<<endl;
//    }
    CalNormalData(merge,20);
    return 0;
}