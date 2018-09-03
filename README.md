# QAapi

将需要预处理的映射数据用pickle保存到stat中，再次运行时从pkl中直接获取

# 项目结构
* service 服务接口
* service_impl 模型加载和预测
    * save_stat 生成映射数组并保存pkl，如果stat不存在就调用
    * load_stat 加载映射
    * classify | ner 预测结果
* util 用到的辅助包
* model.py 合并的两个模型的描述（用于模型加载）

# data 目录结构
``` Plain
├─data
│  aminer_train.dat
│  cleaned_zh_vec
│
├─model
│  classification
│  ner
│
└─stat
   classify_stat.pkl
   ner_stat.pkl
```